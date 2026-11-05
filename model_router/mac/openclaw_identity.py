"""Which OpenClaw agent and channel sent a request.

OpenClaw's provider headers are fixed per provider, but with
compat.sendSessionAffinityHeaders it adds the session id (session_id and
x-session-affinity). This module resolves that id read-only against
OpenClaw's own stores:
  - agents/<id>/agent/openclaw-agent.sqlite: session_nodes maps the session to
    its key (agent:<id>:...), and session_conversations + conversations give
    the channel it last talked on (telegram, whatsapp, ...);
  - openclaw.json and the agent workspace IDENTITY.md give the display name.
Nothing here is trusted for scheduling or security; it only labels jobs.
"""
import json
import re
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

SESSION_ID_RE = re.compile(r'^[\w-]{8,80}$')
CHANNEL_TITLES = {'telegram': 'Telegram', 'whatsapp': 'WhatsApp', 'imessage': 'iMessage', 'slack': 'Slack',
                  'discord': 'Discord', 'signal': 'Signal', 'sms': 'SMS', 'msteams': 'Teams'}


def channel_title(channel):
    return CHANNEL_TITLES.get(channel, channel.replace('-', ' ').title()) if channel else ''


class OpenClawIdentity:
    def __init__(self, home=None, channel_ttl=30.0, names_ttl=300.0):
        self.home = Path(home) if home else Path.home() / '.openclaw'
        self.channel_ttl, self.names_ttl = channel_ttl, names_ttl
        self._agents = {}            # session id -> (agent id, session key); a session never changes agent
        self._channels = {}          # session id -> (checked at, channel)
        self._names, self._names_at = {}, 0.0
        self._lock = threading.Lock()

    def _query(self, agent, sql, args):
        path = self.home / 'agents' / agent / 'agent' / 'openclaw-agent.sqlite'
        if not path.exists(): return None
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=0.5)) as db:
            return db.execute(sql, args).fetchone()

    def _find_agent(self, session_id):
        for directory in sorted((self.home / 'agents').glob('*/agent')):
            agent = directory.parent.name
            row = self._query(agent, 'SELECT session_key FROM session_nodes WHERE current_session_id=? LIMIT 1', (session_id,))
            if row: return agent, row[0]
        return None

    def _channel(self, agent, session_id, session_key):
        row = self._query(agent, 'SELECT c.channel FROM session_conversations s JOIN conversations c USING (conversation_id) '
                                 'WHERE s.session_id=? ORDER BY s.last_seen_at DESC LIMIT 1', (session_id,))
        if row and row[0]: return row[0]
        # Sessions without a delivery conversation still name the dashboard in their key.
        return 'dashboard' if session_key.split(':')[2:3] == ['dashboard'] else ''

    def _name(self, agent):
        if time.time() - self._names_at > self.names_ttl:
            names = {}
            try:
                entries = json.loads((self.home / 'openclaw.json').read_text())['agents']['entries']
            except (OSError, ValueError, KeyError, TypeError):
                entries = {}
            for key, entry in entries.items() if isinstance(entries, dict) else ():
                entry = entry if isinstance(entry, dict) else {}
                name = (entry.get('identity') or {}).get('name') or entry.get('name')
                if not name and entry.get('workspace'):
                    try:
                        found = re.search(r'\*\*Name:\*\*\s*(.+)', (Path(entry['workspace']) / 'IDENTITY.md').read_text())
                        name = found and found[1].strip()
                    except OSError:
                        pass
                names[key] = str(name or key)[:60]
            self._names, self._names_at = names, time.time()
        return self._names.get(agent, agent)

    def lookup(self, session_id):
        """{'agent-id', 'agent_name', 'session-id', 'channel'} for a known session, else {}."""
        if not isinstance(session_id, str): return {}
        session_id = session_id.split(':', 1)[0]  # OpenClaw's prompt cache key is "<session id>:<n>"
        if not SESSION_ID_RE.match(session_id): return {}
        try:
            with self._lock:
                found = self._agents.get(session_id) or self._find_agent(session_id)
                if not found: return {}
                if len(self._agents) > 2048: self._agents.clear()
                self._agents[session_id] = found
                agent, key = found
                checked, channel = self._channels.get(session_id, (0.0, ''))
                if time.time() - checked > self.channel_ttl:
                    channel = self._channel(agent, session_id, key)
                    if len(self._channels) > 2048: self._channels.clear()
                    self._channels[session_id] = (time.time(), channel)
                return {'agent-id': agent, 'agent_name': self._name(agent), 'session-id': session_id, 'channel': channel}
        except sqlite3.Error:
            return {}

import json
import sqlite3

from model_router.mac.app import SHELF_DEFAULTS, job_origin
from model_router.mac.openclaw_identity import OpenClawIdentity


def make_agent(home, agent, sessions):
    db = home / 'agents' / agent / 'agent'
    db.mkdir(parents=True)
    with sqlite3.connect(db / 'openclaw-agent.sqlite') as c:
        c.execute('create table session_nodes (session_key text, current_session_id text)')
        c.execute('create table session_conversations (session_id text, conversation_id text, last_seen_at int)')
        c.execute('create table conversations (conversation_id text, channel text)')
        for key, sid, channels in sessions:
            c.execute('insert into session_nodes values (?,?)', (key, sid))
            for seen, channel in channels:
                c.execute('insert into conversations values (?,?)', (f'{sid}-{seen}', channel))
                c.execute('insert into session_conversations values (?,?,?)', (sid, f'{sid}-{seen}', seen))


def test_lookup_resolves_agent_name_and_latest_channel(tmp_path):
    workspace = tmp_path / 'clawd'; workspace.mkdir()
    (workspace / 'IDENTITY.md').write_text('- **Name:** Mot\n')
    (tmp_path / 'openclaw.json').write_text(json.dumps({'agents': {'entries': {
        'main': {'workspace': str(workspace)}, 'donna': {'name': 'Donna'}, 'chanakya': {'identity': {'name': 'Chanakya'}, 'name': 'x'}}}}))
    make_agent(tmp_path, 'main', [('agent:main:main', 'sess-main-0001', [(1, 'whatsapp'), (2, 'telegram')]),
                                  ('agent:main:dashboard:abc', 'sess-dash-0001', [])])
    make_agent(tmp_path, 'chanakya', [('agent:chanakya:main', 'sess-chan-0001', [(1, 'telegram')])])
    make_agent(tmp_path, 'donna', [('agent:donna:x', 'sess-donn-0001', [])])
    ids = OpenClawIdentity(home=tmp_path)
    assert ids.lookup('sess-main-0001') == {'agent-id': 'main', 'agent_name': 'Mot', 'session-id': 'sess-main-0001', 'channel': 'telegram'}
    assert ids.lookup('sess-chan-0001')['agent_name'] == 'Chanakya'
    assert ids.lookup('sess-main-0001:22')['channel'] == 'telegram'
    assert ids.lookup('sess-dash-0001')['channel'] == 'dashboard'
    assert ids.lookup('sess-donn-0001') == {'agent-id': 'donna', 'agent_name': 'Donna', 'session-id': 'sess-donn-0001', 'channel': ''}
    assert ids.lookup('unknown-session') == {} and ids.lookup("x' or 1=1") == {}


def test_manual_label_names_channel_and_agent():
    policy = SHELF_DEFAULTS
    assert job_origin({'source': '', 'agent-id': 'main', 'agent_name': 'Mot', 'channel': 'telegram'}, {}, policy) == \
        {'origin': 'manual', 'origin_label': 'Telegram · Mot'}
    assert job_origin({'source': '', 'agent-id': 'main', 'agent_name': 'Mot'}, {}, policy)['origin_label'] == 'Manual · Mot'
    # loops keep their run label even when the session has a channel
    loop = job_origin({'source': 'loop', 'agent-id': 'x', 'channel': 'telegram'}, {'loop_id': 'l', 'loop_name': 'Daily'}, policy)
    assert loop['origin_label'] == 'Loop · Daily'

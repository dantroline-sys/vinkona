// BackendClient against canned backend JSON (http MockClient — no server).
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:vinkona_desk/api/backend.dart';

BackendClient _client(Map<String, dynamic> Function(http.Request) route) {
  return BackendClient(
    httpClient: MockClient((req) async => http.Response(
          jsonEncode(route(req)),
          200,
          headers: {'content-type': 'application/json'},
        )),
  );
}

void main() {
  test('activity parses the backend headline', () async {
    final c = _client((req) => {
          'headline': 'Making herself a new tool',
          'doing': 'toolsmith',
          'interruptible': true,
          'session': {'active': false, 'kind': null},
        });
    final a = await c.activity();
    expect(a.headline, 'Making herself a new tool');
    expect(a.doing, 'toolsmith');
    expect(a.sessionActive, isFalse);
  });

  test('feature recipes parse both sides with companion changes', () async {
    final c = _client((req) => {
          'recipes': {
            'research.enabled': {
              'title': 'Researching on her own',
              'enable': {
                'summary': 'She reads up in the background.',
                'changes': [
                  {
                    'path': 'research.idle.enabled',
                    'value': true,
                    'label': 'quiet reading while you are away'
                  }
                ],
                'note': 'News is a separate switch.',
              },
              'disable': {
                'summary': 'She stops reading on her own.',
                'changes': [],
              },
            }
          }
        });
    final rs = await c.featureRecipes();
    expect(rs, hasLength(1));
    expect(rs.first.path, 'research.enabled');
    expect(rs.first.enable.changes.single.path, 'research.idle.enabled');
    expect(rs.first.enable.changes.single.value, true);
    expect(rs.first.disable.changes, isEmpty);
  });

  test('field levels resolve exact, glob, and default', () async {
    final c = _client((req) => {
          'levels': {'tools.own_tools.enabled': 'basic', 'vad.*': 'expert'},
          'default': 'advanced',
          'order': {'basic': 0, 'advanced': 1, 'expert': 2},
          'labels': {},
          'choices': {
            'tools.wikipedia': ['auto', true, false]
          },
        });
    final fl = await c.fieldLevels();
    expect(fl.levelFor('tools.own_tools.enabled'), 'basic');
    expect(fl.levelFor('vad.onset'), 'expert');
    expect(fl.levelFor('anything.else'), 'advanced');
    expect(fl.choices['tools.wikipedia'], ['auto', true, false]);
  });

  test('personas document parses and keeps the raw doc for whole saves',
      () async {
    final c = _client((req) => {
          'default': 'vinkona',
          'personas': {
            'vinkona': {
              'description': 'Warm and witty.',
              'greeting': 'Hey.',
              'voice': 'tara',
              'system_prompt': 'long prompt kept in raw',
            },
            'sage': {'description': 'Calm mentor.', 'voice': 'leo'},
          },
        });
    final doc = await c.personas();
    expect(doc.defaultName, 'vinkona');
    expect(doc.personas.map((p) => p.name), containsAll(['vinkona', 'sage']));
    expect(doc.personas.firstWhere((p) => p.name == 'sage').voice, 'leo');
    // The raw document keeps everything the picker doesn't render.
    expect(
        ((doc.raw['personas'] as Map)['vinkona'] as Map)['system_prompt'],
        'long prompt kept in raw');
  });

  test('tts status parses engines with install state and preset voices',
      () async {
    final c = _client((req) => {
          'current': 'orpheus_gguf',
          'default_voice': 'tara',
          'engines': [
            {
              'key': 'orpheus_gguf',
              'label': 'Orpheus',
              'footprint': 'small',
              'note': 'the default',
              'installed': true,
              'current': true,
              'voices': ['tara', 'leo'],
            },
            {
              'key': 'chatterbox',
              'label': 'Chatterbox',
              'footprint': 'tiny',
              'note': 'clones a clip',
              'installed': false,
              'current': false,
              'voices': [],
            },
          ],
        });
    final t = await c.tts();
    expect(t.current, 'orpheus_gguf');
    expect(t.currentEngine?.voices, ['tara', 'leo']);
    expect(t.engines[1].installed, isFalse);
  });

  test('profiles parse with stats and the active flag', () async {
    final c = _client((req) => {
          'active': 'default',
          'profiles': [
            {
              'name': 'default',
              'memories': 42,
              'personas': 3,
              'size': 2048,
              'active': true
            },
            {
              'name': 'guest',
              'memories': -1,
              'personas': 0,
              'size': 0,
              'active': false
            },
          ],
        });
    final s = await c.profiles();
    expect(s.active, 'default');
    expect(s.profiles.first.memories, 42);
    expect(s.profiles.last.active, isFalse);
  });

  test('a non-200 surfaces as a BackendException', () async {
    final c = BackendClient(
        httpClient: MockClient((req) async => http.Response('nope', 500)));
    expect(c.activity(), throwsA(isA<BackendException>()));
  });

  group('dotted config paths', () {
    test('get walks nested maps and misses safely', () {
      final cfg = {
        'tools': {
          'own_tools': {'enabled': true}
        }
      };
      expect(configGet(cfg, 'tools.own_tools.enabled'), true);
      expect(configGet(cfg, 'tools.missing.x'), isNull);
    });

    test('set writes deep and creates intermediate maps', () {
      final cfg = <String, dynamic>{};
      configSet(cfg, 'a.b.c', 5);
      expect(configGet(cfg, 'a.b.c'), 5);
      configSet(cfg, 'a.b.c', 6);
      expect(configGet(cfg, 'a.b.c'), 6);
    });
  });
}

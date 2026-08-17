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
        });
    final fl = await c.fieldLevels();
    expect(fl.levelFor('tools.own_tools.enabled'), 'basic');
    expect(fl.levelFor('vad.onset'), 'expert');
    expect(fl.levelFor('anything.else'), 'advanced');
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

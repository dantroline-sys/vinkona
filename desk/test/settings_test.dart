// The Basic settings surface end-to-end against a mock backend: recipes render
// as switches, the flip shows the preview sheet, confirming posts the switch
// AND its companion changes in one whole-config save.
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:vinkona_desk/api/backend.dart';
import 'package:vinkona_desk/screens/settings_basic.dart';

void main() {
  testWidgets('flip → preview → confirm saves switch + companions',
      (tester) async {
    Map<String, dynamic>? saved;
    final client = BackendClient(
      httpClient: MockClient((req) async {
        if (req.url.path == '/api/config' && req.method == 'GET') {
          return http.Response(
              jsonEncode({
                'research': {
                  'enabled': false,
                  'idle': {'enabled': false},
                }
              }),
              200);
        }
        if (req.url.path == '/api/config' && req.method == 'POST') {
          saved = jsonDecode(req.body) as Map<String, dynamic>;
          return http.Response('{"ok": true}', 200);
        }
        if (req.url.path == '/api/feature_recipes') {
          return http.Response(
              jsonEncode({
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
                    },
                    'disable': {'summary': 'She stops.', 'changes': []},
                  }
                }
              }),
              200);
        }
        return http.Response('{}', 200);
      }),
    );

    await tester.pumpWidget(
        MaterialApp(home: Scaffold(body: SettingsBasicScreen(client: client))));
    await tester.pumpAndSettle();

    // The recipe renders as a switch, currently off.
    expect(find.text('Researching on her own'), findsOneWidget);
    final sw = tester.widget<SwitchListTile>(find.byType(SwitchListTile));
    expect(sw.value, isFalse);

    // Flip it: the preview sheet appears with the companion change, unsaved.
    await tester.tap(find.byType(SwitchListTile));
    await tester.pumpAndSettle();
    expect(find.text('Turn on: Researching on her own'), findsOneWidget);
    expect(find.text('quiet reading while you are away'), findsOneWidget);
    expect(saved, isNull, reason: 'nothing saves before the confirm');

    // Confirm: the switch AND its companion land in one whole-config save.
    await tester.tap(find.widgetWithText(FilledButton, 'Turn on'));
    await tester.pumpAndSettle();
    expect(saved, isNotNull);
    expect(configGet(saved!, 'research.enabled'), true);
    expect(configGet(saved!, 'research.idle.enabled'), true);
  });

  testWidgets('cancelling the preview changes nothing', (tester) async {
    var posted = false;
    final client = BackendClient(
      httpClient: MockClient((req) async {
        if (req.url.path == '/api/config' && req.method == 'POST') {
          posted = true;
          return http.Response('{"ok": true}', 200);
        }
        if (req.url.path == '/api/config') {
          return http.Response(
              jsonEncode({
                'research': {'enabled': false}
              }),
              200);
        }
        if (req.url.path == '/api/feature_recipes') {
          return http.Response(
              jsonEncode({
                'recipes': {
                  'research.enabled': {
                    'title': 'Researching on her own',
                    'enable': {'summary': 'On.', 'changes': []},
                    'disable': {'summary': 'Off.', 'changes': []},
                  }
                }
              }),
              200);
        }
        return http.Response('{}', 200);
      }),
    );

    await tester.pumpWidget(
        MaterialApp(home: Scaffold(body: SettingsBasicScreen(client: client))));
    await tester.pumpAndSettle();
    await tester.tap(find.byType(SwitchListTile));
    await tester.pumpAndSettle();
    await tester.tap(find.widgetWithText(TextButton, 'Cancel'));
    await tester.pumpAndSettle();
    expect(posted, isFalse);
    expect(tester.widget<SwitchListTile>(find.byType(SwitchListTile)).value,
        isFalse);
  });
}

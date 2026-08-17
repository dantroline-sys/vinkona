// The Everyday surface: the field list is server-derived (Basic tier minus
// recipe switches minus persona/voice), each value type gets the right
// control, and every edit saves the whole config.
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:vinkona_desk/api/backend.dart';
import 'package:vinkona_desk/screens/everyday.dart';

BackendClient _mock({void Function(Map<String, dynamic>)? onConfigPost}) {
  return BackendClient(
    httpClient: MockClient((req) async {
      final path = req.url.path;
      if (path == '/api/config' && req.method == 'POST') {
        onConfigPost?.call(jsonDecode(req.body) as Map<String, dynamic>);
        return http.Response('{"ok": true}', 200);
      }
      if (path == '/api/config') {
        return http.Response(
            jsonEncode({
              'asides': {'enabled': true},
              'awareness': {'location': ''},
              'tools': {'wikipedia': 'auto'},
              'notifications': {
                'lead_times_min': [1440, 60]
              },
              'research': {'enabled': false},
            }),
            200);
      }
      if (path == '/api/field_levels') {
        return http.Response(
            jsonEncode({
              'levels': {
                'asides.enabled': 'basic',
                'awareness.location': 'basic',
                'tools.wikipedia': 'basic',
                'notifications.lead_times_min': 'basic',
                'research.enabled': 'basic', // covered by a recipe → excluded
              },
              'default': 'advanced',
              'order': {'basic': 0, 'advanced': 1, 'expert': 2},
              'labels': {
                'asides.enabled': 'Private asides',
                'tools.wikipedia': 'Wikipedia lookup',
                'awareness.location': 'Where you are',
                'notifications.lead_times_min': 'Reminder lead times (minutes)',
              },
              'choices': {
                'tools.wikipedia': ['auto', true, false]
              },
            }),
            200);
      }
      if (path == '/api/feature_recipes') {
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
      return http.Response('{"help": {}}', 200);
    }),
  );
}

Future<void> _pump(WidgetTester tester, BackendClient client) async {
  await tester
      .pumpWidget(MaterialApp(home: Scaffold(body: EverydayScreen(client: client))));
  await tester.pumpAndSettle();
}

void main() {
  testWidgets('renders Basic fields but not recipe-covered ones',
      (tester) async {
    await _pump(tester, _mock());
    expect(find.text('Private asides'), findsOneWidget);
    expect(find.text('Wikipedia lookup'), findsOneWidget);
    expect(find.text('Where you are'), findsOneWidget);
    // research.enabled is Basic but belongs to the Features tab's recipe —
    // its fallback label must not appear here.
    expect(find.text('Enabled'), findsNothing);
    expect(find.text('Researching on her own'), findsNothing);
  });

  testWidgets('a boolean switch saves the whole config with the flip',
      (tester) async {
    Map<String, dynamic>? saved;
    await _pump(tester, _mock(onConfigPost: (b) => saved = b));

    await tester.tap(find.widgetWithText(SwitchListTile, 'Private asides'));
    await tester.pumpAndSettle();
    expect(saved, isNotNull);
    expect(configGet(saved!, 'asides.enabled'), false);
    // The rest of the config travelled with it, untouched.
    expect(configGet(saved!, 'research.enabled'), false);
    expect(configGet(saved!, 'tools.wikipedia'), 'auto');
  });

  testWidgets('a fixed-choice field saves the chosen value (mixed types)',
      (tester) async {
    Map<String, dynamic>? saved;
    await _pump(tester, _mock(onConfigPost: (b) => saved = b));

    await tester.tap(find.text('Off'));
    await tester.pumpAndSettle();
    expect(saved, isNotNull);
    expect(configGet(saved!, 'tools.wikipedia'), false);
  });

  testWidgets('a number-list field parses comma text back to integers',
      (tester) async {
    Map<String, dynamic>? saved;
    await _pump(tester, _mock(onConfigPost: (b) => saved = b));

    await tester.enterText(find.widgetWithText(TextField, '1440, 60'), '60, 30');
    await tester.testTextInput.receiveAction(TextInputAction.done);
    await tester.pumpAndSettle();
    expect(saved, isNotNull);
    expect(configGet(saved!, 'notifications.lead_times_min'), [60, 30]);
  });
}

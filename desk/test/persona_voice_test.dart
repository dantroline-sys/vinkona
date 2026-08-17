// Persona & voice against a mock backend: picking a persona posts the WHOLE
// personas document with the new default; switching the engine goes through
// the confirm sheet to /api/tts/select; a voice chip saves tts.default_voice
// via the whole-config contract.
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:vinkona_desk/api/backend.dart';
import 'package:vinkona_desk/screens/persona_voice.dart';

BackendClient _mock({
  void Function(Map<String, dynamic>)? onPersonasPost,
  void Function(Map<String, dynamic>)? onTtsSelect,
  void Function(Map<String, dynamic>)? onConfigPost,
}) {
  return BackendClient(
    httpClient: MockClient((req) async {
      final path = req.url.path;
      if (path == '/api/personas' && req.method == 'POST') {
        onPersonasPost?.call(jsonDecode(req.body) as Map<String, dynamic>);
        return http.Response('{}', 200);
      }
      if (path == '/api/personas') {
        return http.Response(
            jsonEncode({
              'default': 'vinkona',
              'personas': {
                'vinkona': {
                  'description': 'Warm and witty.',
                  'voice': 'tara',
                  'system_prompt': 'kept whole',
                },
                'sage': {'description': 'Calm mentor.', 'voice': 'leo'},
              },
            }),
            200);
      }
      if (path == '/api/tts/select') {
        onTtsSelect?.call(jsonDecode(req.body) as Map<String, dynamic>);
        return http.Response('{"ok": true, "scheduled": true}', 200);
      }
      if (path == '/api/tts') {
        return http.Response(
            jsonEncode({
              'current': 'orpheus_gguf',
              'default_voice': 'tara',
              'engines': [
                {
                  'key': 'orpheus_gguf',
                  'label': 'Orpheus',
                  'footprint': 'small',
                  'note': 'The tuned default.',
                  'installed': true,
                  'current': true,
                  'voices': ['tara', 'leo'],
                },
                {
                  'key': 'chatterbox',
                  'label': 'Chatterbox',
                  'footprint': 'tiny',
                  'note': 'Clones a clip.',
                  'installed': false,
                  'current': false,
                  'voices': [],
                },
              ],
            }),
            200);
      }
      if (path == '/api/config' && req.method == 'POST') {
        onConfigPost?.call(jsonDecode(req.body) as Map<String, dynamic>);
        return http.Response('{"ok": true}', 200);
      }
      if (path == '/api/config') {
        return http.Response(
            jsonEncode({
              'tts': {'default_voice': 'tara'}
            }),
            200);
      }
      return http.Response('{"help": {}}', 200);
    }),
  );
}

Future<void> _pump(WidgetTester tester, BackendClient client) async {
  await tester.pumpWidget(MaterialApp(
      home: Scaffold(body: PersonaVoiceScreen(client: client))));
  await tester.pumpAndSettle();
}

void main() {
  testWidgets('picking a persona posts the whole document with a new default',
      (tester) async {
    Map<String, dynamic>? posted;
    await _pump(tester, _mock(onPersonasPost: (b) => posted = b));

    expect(find.text('Vinkona'), findsOneWidget);
    await tester.tap(find.text('Sage'));
    await tester.pumpAndSettle();

    expect(posted, isNotNull);
    expect(posted!['default'], 'sage');
    // The rest of the document travelled untouched (prompts and all).
    expect(((posted!['personas'] as Map)['vinkona'] as Map)['system_prompt'],
        'kept whole');
  });

  testWidgets('switching the engine confirms first, then posts tts/select',
      (tester) async {
    Map<String, dynamic>? selected;
    await _pump(tester, _mock(onTtsSelect: (b) => selected = b));

    await tester.tap(find.text('Chatterbox'));
    await tester.pumpAndSettle();
    // The sheet explains the switch, flags the missing install, and nothing
    // has been posted yet.
    expect(find.text('Switch her voice to Chatterbox'), findsOneWidget);
    expect(find.textContaining("isn't set up yet"), findsOneWidget);
    expect(selected, isNull);

    await tester.tap(find.text('At the next restart'));
    await tester.pumpAndSettle();
    expect(selected, isNotNull);
    expect(selected!['engine'], 'chatterbox');
    expect(selected!['when'], 'next_restart');
  });

  testWidgets('a voice chip saves tts.default_voice via the whole config',
      (tester) async {
    Map<String, dynamic>? saved;
    await _pump(tester, _mock(onConfigPost: (b) => saved = b));

    // The chips sit at the bottom of a lazy ListView — scroll them into being.
    await tester.scrollUntilVisible(
        find.widgetWithText(ChoiceChip, 'leo'), 200);
    await tester.tap(find.widgetWithText(ChoiceChip, 'leo'));
    await tester.pumpAndSettle();
    expect(saved, isNotNull);
    expect(configGet(saved!, 'tts.default_voice'), 'leo');
  });
}

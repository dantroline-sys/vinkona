// Basic smoke test: the app builds and shows the Connect button.

import 'package:flutter_test/flutter_test.dart';

import 'package:vinkona_client/main.dart';

void main() {
  testWidgets('App builds and shows Connect button', (WidgetTester tester) async {
    await tester.pumpWidget(const VinkonaApp());

    expect(find.text('Vinkona'), findsOneWidget);
    expect(find.text('Connect'), findsOneWidget);
  });
}

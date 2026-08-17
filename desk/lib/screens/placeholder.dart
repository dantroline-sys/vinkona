/// A stub screen for surfaces that land in a later stage of DESK.md — honest
/// about what's coming rather than an empty pane.
library;

import 'package:flutter/material.dart';

class ComingSoonScreen extends StatelessWidget {
  final String title;
  final String detail;
  const ComingSoonScreen({super.key, required this.title, required this.detail});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 420),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.construction,
                size: 40, color: Theme.of(context).colorScheme.outline),
            const SizedBox(height: 12),
            Text(title, style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 8),
            Text(detail, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}

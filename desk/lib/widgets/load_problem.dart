// The friendly can't-reach-her card, shared by every settings surface.  First
// run and not-running look the same from a thin client, so the card covers
// both: start her, or install her — never a stack trace.
import 'package:flutter/material.dart';

class LoadProblem extends StatelessWidget {
  final String detail; // the underlying error, shown small for the curious
  final VoidCallback onRetry;
  const LoadProblem({super.key, required this.detail, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 460),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.power_off_outlined,
                size: 40, color: Theme.of(context).colorScheme.outline),
            const SizedBox(height: 12),
            Text("Can't reach Vinkona",
                style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 8),
            const Text(
              'She may not be running yet.  Start her with ./vinkona.sh in '
              'her folder — or, if she is not installed on this computer, '
              'the install guide (README.md) walks you through it.',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(detail,
                textAlign: TextAlign.center,
                style: Theme.of(context)
                    .textTheme
                    .bodySmall
                    ?.copyWith(color: Theme.of(context).colorScheme.outline)),
            const SizedBox(height: 16),
            FilledButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh),
                label: const Text('Try again')),
          ],
        ),
      ),
    );
  }
}

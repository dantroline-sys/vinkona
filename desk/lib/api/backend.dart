/// The one seam to Vinkona's backend: a thin, typed client over the config
/// panel's localhost HTTP API (config_server.py).
///
/// THE RULE OF THIS APP: no logic lives here in Dart.  The backend owns every
/// decision (field audience levels, feature recipes, activity resolution); this
/// client fetches, displays, and posts back.  Anything smarter belongs in
/// config_server, where the web panel and the phone client can share it.
library;

import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

/// One plain-language companion change inside a feature recipe.
class RecipeChange {
  final String path;
  final Object? value;
  final String label;
  RecipeChange({required this.path, required this.value, required this.label});

  factory RecipeChange.fromJson(Map<String, dynamic> j) => RecipeChange(
        path: j['path'] as String? ?? '',
        value: j['value'],
        label: j['label'] as String? ?? '',
      );
}

/// One direction (enable/disable) of a feature recipe: what flipping the
/// switch means, in plain words, and the companion settings that follow it.
class RecipeSide {
  final String summary;
  final String note;
  final List<RecipeChange> changes;
  RecipeSide({required this.summary, required this.note, required this.changes});

  factory RecipeSide.fromJson(Map<String, dynamic>? j) => RecipeSide(
        summary: j?['summary'] as String? ?? '',
        note: j?['note'] as String? ?? '',
        changes: ((j?['changes'] as List?) ?? const [])
            .whereType<Map<String, dynamic>>()
            .map(RecipeChange.fromJson)
            .toList(),
      );
}

/// A Basic-tier feature toggle with its preview-then-confirm story.
class FeatureRecipe {
  final String path; // the dotted config path of the on/off switch
  final String title;
  final RecipeSide enable;
  final RecipeSide disable;
  FeatureRecipe(
      {required this.path,
      required this.title,
      required this.enable,
      required this.disable});

  factory FeatureRecipe.fromJson(String path, Map<String, dynamic> j) =>
      FeatureRecipe(
        path: path,
        title: j['title'] as String? ?? path,
        enable: RecipeSide.fromJson(j['enable'] as Map<String, dynamic>?),
        disable: RecipeSide.fromJson(j['disable'] as Map<String, dynamic>?),
      );
}

/// What Vinkona is doing right now (the backend resolves the single headline).
class Activity {
  final String headline;
  final String doing;
  final bool interruptible;
  final bool sessionActive;
  Activity(
      {required this.headline,
      required this.doing,
      required this.interruptible,
      required this.sessionActive});

  factory Activity.fromJson(Map<String, dynamic> j) => Activity(
        headline: j['headline'] as String? ?? 'Unknown',
        doing: j['doing'] as String? ?? '',
        interruptible: j['interruptible'] as bool? ?? true,
        sessionActive:
            (j['session'] as Map<String, dynamic>?)?['active'] as bool? ?? false,
      );
}

/// The audience-tier map: dotted config path -> basic | advanced | expert.
class FieldLevels {
  final Map<String, String> levels;
  final String defaultLevel;
  final Map<String, String> labels; // optional friendly names per path
  FieldLevels(
      {required this.levels, required this.defaultLevel, required this.labels});

  String levelFor(String path) {
    // Exact match wins; then glob prefixes ("vad.*"); then the default.
    final exact = levels[path];
    if (exact != null) return exact;
    for (final e in levels.entries) {
      if (e.key.endsWith('.*') &&
          path.startsWith(e.key.substring(0, e.key.length - 1))) {
        return e.value;
      }
    }
    return defaultLevel;
  }

  factory FieldLevels.fromJson(Map<String, dynamic> j) => FieldLevels(
        levels: ((j['levels'] as Map<String, dynamic>?) ?? const {})
            .map((k, v) => MapEntry(k, v as String)),
        defaultLevel: j['default'] as String? ?? 'advanced',
        labels: ((j['labels'] as Map<String, dynamic>?) ?? const {})
            .map((k, v) => MapEntry(k, v as String)),
      );
}

class BackendException implements Exception {
  final String message;
  BackendException(this.message);
  @override
  String toString() => message;
}

/// The client.  Constructed once; screens call and render.
class BackendClient {
  final String baseUrl;
  final http.Client _http;
  BackendClient({this.baseUrl = 'http://127.0.0.1:8090', http.Client? httpClient})
      : _http = httpClient ?? http.Client();

  Future<Map<String, dynamic>> _getJson(String path) async {
    final r = await _http
        .get(Uri.parse('$baseUrl$path'))
        .timeout(const Duration(seconds: 6));
    if (r.statusCode != 200) {
      throw BackendException('HTTP ${r.statusCode} from $path');
    }
    final body = jsonDecode(r.body);
    if (body is! Map<String, dynamic>) {
      throw BackendException('$path returned a non-object');
    }
    return body;
  }

  /// The merged config (defaults ⊕ the user's overrides): what every settings
  /// surface edits.  Saved back WHOLE via [saveConfig] — same contract as the
  /// web panel.
  Future<Map<String, dynamic>> config() => _getJson('/api/config');

  Future<void> saveConfig(Map<String, dynamic> full) async {
    final r = await _http
        .post(Uri.parse('$baseUrl/api/config'), body: jsonEncode(full))
        .timeout(const Duration(seconds: 10));
    if (r.statusCode != 200) {
      throw BackendException('config save failed: HTTP ${r.statusCode}');
    }
  }

  Future<FieldLevels> fieldLevels() async =>
      FieldLevels.fromJson(await _getJson('/api/field_levels'));

  Future<List<FeatureRecipe>> featureRecipes() async {
    final j = await _getJson('/api/feature_recipes');
    final recipes = (j['recipes'] as Map<String, dynamic>?) ?? const {};
    return recipes.entries
        .where((e) => e.value is Map<String, dynamic>)
        .map((e) =>
            FeatureRecipe.fromJson(e.key, e.value as Map<String, dynamic>))
        .toList();
  }

  Future<Activity> activity() async =>
      Activity.fromJson(await _getJson('/api/activity'));

  /// Help text per dotted config path (extracted from config.py's comments).
  Future<Map<String, String>> help() async {
    final j = await _getJson('/api/help');
    final h = (j['help'] as Map<String, dynamic>?) ?? const {};
    return h.map((k, v) => MapEntry(k, v.toString()));
  }

  /// True when the backend answers at all — the connection card's probe.
  Future<bool> reachable() async {
    try {
      await activity();
      return true;
    } catch (_) {
      return false;
    }
  }

  void close() => _http.close();
}

/// Read or write a dotted path inside the (mutable) merged-config map.
Object? configGet(Map<String, dynamic> cfg, String dotted) {
  Object? cur = cfg;
  for (final part in dotted.split('.')) {
    if (cur is Map<String, dynamic>) {
      cur = cur[part];
    } else {
      return null;
    }
  }
  return cur;
}

void configSet(Map<String, dynamic> cfg, String dotted, Object? value) {
  final parts = dotted.split('.');
  Map<String, dynamic> cur = cfg;
  for (final part in parts.sublist(0, parts.length - 1)) {
    final next = cur[part];
    if (next is Map<String, dynamic>) {
      cur = next;
    } else {
      final made = <String, dynamic>{};
      cur[part] = made;
      cur = made;
    }
  }
  cur[parts.last] = value;
}

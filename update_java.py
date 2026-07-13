import re

# 1. SearchResult.java
sr_path = "turboquant-explained/android/TurboQuantBench/turbovec-lib/src/main/java/com/turbovec/lib/SearchResult.java"
with open(sr_path, "r") as f:
    sr_content = f.read()

sr_content = sr_content.replace("public final long[] ids;", "public final int[] ids;")
sr_content = sr_content.replace("SearchResult(float[] scores, long[] ids, int count)", "SearchResult(float[] scores, int[] ids, int count)")
sr_content = sr_content.replace("long IDs", "int IDs")

with open(sr_path, "w") as f:
    f.write(sr_content)

# 2. TurboVecIndex.java
tvi_path = "turboquant-explained/android/TurboQuantBench/turbovec-lib/src/main/java/com/turbovec/lib/TurboVecIndex.java"
with open(tvi_path, "r") as f:
    tvi_content = f.read()

tvi_content = tvi_content.replace("long[] ids", "int[] ids")
tvi_content = tvi_content.replace("long[] outIds", "int[] outIds")
tvi_content = tvi_content.replace("boolean remove(long id)", "boolean remove(int id)")
tvi_content = tvi_content.replace("boolean nativeIdMapRemove(long handle, long id)", "boolean nativeIdMapRemove(long handle, int id)")
tvi_content = tvi_content.replace("boolean contains(long id)", "boolean contains(int id)")
tvi_content = tvi_content.replace("boolean nativeIdMapContains(long handle, long id)", "boolean nativeIdMapContains(long handle, int id)")

with open(tvi_path, "w") as f:
    f.write(tvi_content)

# 3. MainActivity.java in sample-app
ma_path = "turboquant-explained/android/TurboQuantBench/sample-app/src/main/java/com/turbovec/sample/MainActivity.java"
with open(ma_path, "r") as f:
    ma_content = f.read()

ma_content = ma_content.replace("long[] TEST_IDS = { 101L, 102L, 103L, 104L, 105L };", "int[] TEST_IDS = { 101, 102, 103, 104, 105 };")
ma_content = ma_content.replace("long[] outIds = new long[SEARCH_K];", "int[] outIds = new int[SEARCH_K];")
ma_content = ma_content.replace("long[] ids = outIds.clone();", "int[] ids = outIds.clone();")
ma_content = ma_content.replace("for (long id : TEST_IDS)", "for (int id : TEST_IDS)")
ma_content = ma_content.replace("removeAll(new long[]{104L, 999L})", "removeAll(new int[]{104, 999})")
ma_content = ma_content.replace("showResultsCard(int count, float[] scores, long[] ids)", "showResultsCard(int count, float[] scores, int[] ids)")
ma_content = ma_content.replace("add(float[], dim, long[] ids)", "add(float[], dim, int[] ids)")
ma_content = ma_content.replace("search(queries, k, float[] scores, long[] ids)", "search(queries, k, float[] scores, int[] ids)")
ma_content = ma_content.replace("removeAll(long[])", "removeAll(int[])")

with open(ma_path, "w") as f:
    f.write(ma_content)

# 4. native/src/lib.rs
lib_path = "turboquant-explained/android/TurboQuantBench/native/src/lib.rs"
with open(lib_path, "r") as f:
    lib_content = f.read()

# Replace IdMapIndex with IdMapIndex32 everywhere in the JNI methods
# First, remove the previously appended block of IdMapIndex32 JNI
idx = lib_content.find("// ─── IdMapIndex32 JNI ────────────────────────────────────────────────────────")
if idx != -1:
    lib_content = lib_content[:idx]

# Replace IdMapIndex -> turbovec::IdMapIndex32 in id_map_mut, id_map_ref, and native methods
lib_content = lib_content.replace(" turbovec::IdMapIndex", " turbovec::IdMapIndex32")
# Wait, it might be `turbovec::IdMapIndex` or `IdMapIndex` imported.
# Let's check imports: `use turbovec::{IdMapIndex, SearchResult, TurboQuantIndex};`
# Change it to `use turbovec::{IdMapIndex32, SearchResult, TurboQuantIndex};`
lib_content = lib_content.replace("use turbovec::{IdMapIndex, SearchResult, TurboQuantIndex};", "use turbovec::{IdMapIndex32, SearchResult, TurboQuantIndex};")

# Replace IdMapIndex with IdMapIndex32 everywhere
lib_content = lib_content.replace(" IdMapIndex", " IdMapIndex32")
lib_content = lib_content.replace("<IdMapIndex>", "<IdMapIndex32>")

# Change nativeIdMapAdd
lib_content = lib_content.replace("ids: JLongArray<'_>", "ids: JIntArray<'_>")
lib_content = lib_content.replace("read_long_array(&mut env, &ids)", "read_int_array(&mut env, &ids)")

# Change nativeIdMapSearch
lib_content = lib_content.replace("out_ids: JLongArray<'_>", "out_ids: JIntArray<'_>")
lib_content = lib_content.replace("ids: Vec<u64> = ids.into_iter().map(|id| id as u64).collect();", "ids: Vec<i32> = ids.into_iter().map(|id| id as i32).collect();")
lib_content = lib_content.replace("ids_i64: Vec<i64> = ids.into_iter().map(|id| id as i64).collect();", "ids_i32: Vec<i32> = ids.into_iter().map(|id| id as i32).collect();")
lib_content = lib_content.replace("set_long_array_region(&out_ids, 0, &ids_i64)", "set_int_array_region(&out_ids, 0, &ids_i32)")
# SearchResult overload
lib_content = lib_content.replace("set_long_array_region(&out_ids, 0, &ids)", "set_int_array_region(&out_ids, 0, &ids_i32)")

# Change nativeIdMapRemove and nativeIdMapContains
lib_content = lib_content.replace("id: jlong", "id: jint")
lib_content = lib_content.replace("id as u64", "id as u32")

# Also in nativeIdMapSearchResult:
# `let ids_i64: Vec<i64> = ids.into_iter().map(|id| id as i64).collect();`
# And `let out_ids = match env.new_long_array(count as jsize) { ... env.set_long_array_region(&out_ids, 0, &ids_i64)` -> `new_int_array`, `set_int_array_region`, etc.

# More precise string replacements for SearchResult:
lib_content = lib_content.replace("new_long_array(count", "new_int_array(count")
lib_content = lib_content.replace("([F[JI)V", "([F[II)V") # Signature for SearchResult constructor

with open(lib_path, "w") as f:
    f.write(lib_content)


# TurboQuant Explained

Static GitHub Pages explainer plus an Android benchmark app for TurboQuant vector search.

Live site after Pages deploy:

https://ravisankarg.github.io/turboquant-explained/

## What Is In This Repo

- `index.html`, `styles.css`, `script.js`: the GitHub Pages explainer with visualizations, benchmark tables, and Android app documentation.
- `turbovec/`: cloned and extended Rust TurboVec/TurboQuant implementation.
- `android/TurboQuantBench/`: Android app that calls the Rust search/index code through JNI.
- `releases/android/TurboQuantBench-release.apk`: release APK tested on Samsung Galaxy S25 Ultra.
- `releases/android/turbovec-lib-release.aar`: pre-built Android library AAR for embedding TurboVecIndex in your own app.
- `docs/ANDROID_APP.md`: app usage, build, and implementation notes.
- `docs/BENCHMARK_RESULTS.md`: S25 Ultra benchmark results and KPI definitions.

## Android Benchmark Flow

The app does this on-device:

1. Download either the first 50,000 or first 1,000,000 Cohere 768-d vectors from Hugging Face raw `.f32` data.
2. Downloads continue in a foreground data-sync service if the app is backgrounded or the screen turns off. Interrupted transfers resume from the saved `.part` byte offset.
3. Tap the benchmark button; it detects every downloaded dataset and runs in a foreground service, so the job continues if the app is backgrounded or the screen turns off.
4. For each available dataset, build an exact FP32 baseline for recall ground truth.
5. Build the bundled HNSW graph baseline and TurboQuant indexes at 8, 4, 3, and 2 bits.
6. Run 1000 self queries and 1000 deterministic random mixture queries per dataset.
7. Report one combined KPI table with dataset, vector count, R@1, R@10, index time, prepare time, search latency, ROM, and RAM. Raw f32 database staging is capped at 100 MB while building/searching benchmark methods.

Install the tested APK:

```bash
adb install -r releases/android/TurboQuantBench-release.apk
```

Build locally:

```bash
cd android/TurboQuantBench
cp local.properties.example local.properties
# edit sdk.dir if needed
JAVA_HOME=/home/ravi/AG/Android_SDK/jdk \
PATH=/home/ravi/AG/Android_SDK/jdk/bin:/home/ravi/AG/Android_SDK/gradle/bin:$HOME/.cargo/bin:$PATH \
gradle assembleRelease
```

If `app/release.keystore` is absent, the Gradle build falls back to debug signing for local reproducibility.

## S25 Ultra Results

Measured dataset: first 50K vectors from `YoKONCy/Cohere-1M-wikipedia-768d`, 768 dimensions. The app can now also download and benchmark the first 1M vectors when enough storage and runtime are available on the phone.

| Method | Bits | Self R@1 | Self R@10 | Random R@1 | Random R@10 | Index ms | Prep ms | Write ms | Self ms | Random ms | us/query | ROM | RAM delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact fp32 | 32 | 100.00% | 100.00% | 100.00% | 100.00% | 0.0 | 0.0 | 0.0 | 4155.7 | 4155.7 | 4155.7 | 146.5 MB | 146.5 MB |
| turbovec | 8 | 100.00% | 100.00% | 100.00% | 100.00% | 1293.7 | 259.4 | 17.2 | 4886.6 | 4890.9 | 4888.8 | 36.8 MB | 124.4 MB |
| turbovec | 4 | 100.00% | 100.00% | 99.20% | 100.00% | 994.9 | 147.4 | 8.1 | 143.6 | 149.8 | 146.7 | 18.5 MB | 78.5 MB |
| turbovec | 3 | 100.00% | 100.00% | 99.20% | 100.00% | 977.0 | 74.0 | 6.0 | 148.0 | 153.0 | 150.5 | 13.9 MB | 34.4 MB |
| turbovec | 2 | 99.60% | 100.00% | 89.70% | 99.80% | 927.1 | 79.5 | 4.3 | 83.2 | 80.4 | 81.8 | 9.4 MB | 37.6 MB |

`us/query` is the warm query latency after `prepare()` has already built search caches.

## Rust Changes

The cloned TurboVec code was extended to support 8-bit indexes:

- `turbovec/turbovec/src/lib.rs`: constructors accept `bit_width=8`.
- `turbovec/turbovec/src/io.rs`: persisted `.tv` / `.tvim` load validation accepts 8-bit.
- `turbovec/turbovec/src/encode.rs`: 8-bit quantization uses binary search over Lloyd-Max boundaries.
- `turbovec/turbovec/src/search.rs`: Android ARM path has a block-major 8-bit byte scorer with NEON-built query-centroid LUTs.

Low-bit 2/3/4 search keeps the original optimized NEON nibble-LUT path.

## Why Native Is Used

The fast paths are in Rust/JNI, not Java:

- One foreground service enters native code for the full benchmark, so the run survives backgrounding and screen-off periods.
- Rayon parallelism is used for FP32 baseline and exact truth generation.
- HNSW is bundled into the same native library via `hnsw_rs`; it does not require any separate install on the phone.
- TurboQuant search scans a blocked 32-vector layout prepared once by `prepare()`.
- 2/3/4-bit search uses ARM NEON lookup-table kernels.
- 8-bit search uses exact byte-code centroid scoring with a NEON-precomputed query LUT.

Java handles UI, download progress, and table rendering only.

---

## TurboVec Android Library API (`com.turbovec.lib`)

A high-performance Android library AAR containing the quantized vector index `TurboVecIndex` with stable external `long` IDs.

### Installation

Copy the release AAR from [releases/android/turbovec-lib-release.aar](releases/android/turbovec-lib-release.aar) into your project's `libs/` folder and declare the dependency in `build.gradle`:

```groovy
dependencies {
    implementation files('libs/turbovec-lib-release.aar')
}
```

Ensure your app targets the `arm64-v8a` ABI:

```groovy
android {
    defaultConfig {
        ndk {
            abiFilters "arm64-v8a"
        }
    }
}
```

### Build the AAR yourself

```bash
cd android/TurboQuantBench
.\gradlew.bat :turbovec-lib:assembleRelease
# output: turbovec-lib/build/outputs/aar/turbovec-lib-release.aar
```

### API Usage Example

```java
import com.turbovec.lib.TurboVecIndex;
import com.turbovec.lib.SearchResult;

String indexPath = context.getFilesDir() + "/vectors.tvim";

try (TurboVecIndex index = new TurboVecIndex(384, 4, indexPath)) {

    // Add vectors with stable external IDs
    float[] vectors = new float[] { /* 5 x 384 flat floats */ };
    index.add(vectors, 384, new long[]{ 101L, 102L, 103L, 104L, 105L });

    // Warm search caches — recommended after add() or load()
    index.prepare();

    float[] query = new float[] { /* 384-d query vector */ };

    // Option A: SearchResult API — convenient, allocates arrays internally
    SearchResult result = index.search(query, 3);
    if (!result.isEmpty()) {
        for (int i = 0; i < result.count; i++)
            Log.d("TurboVec", "rank=" + i + " id=" + result.ids[i] + " score=" + result.scores[i]);
    }
    Log.d("TurboVec", result.toDebugString(5));
    // -> SearchResult{count=3, top=[(id=102, score=0.9821), (id=101, score=0.9743), ...]}

    // Option B: Output-parameter API — zero-allocation, reuse buffers in hot paths
    float[] outScores = new float[3];
    long[]  outIds    = new long[3];
    int n = index.search(query, 3, outScores, outIds);

    // Delete and persist
    index.remove(103L);
    index.removeAll(new long[]{ 104L, 105L });
    index.save();
} catch (IOException e) { e.printStackTrace(); }
```

### Lifecycle: Load → Prepare → Search

When loading a saved index, **always call `prepare()` before searching**:

```java
// Correct
TurboVecIndex index = TurboVecIndex.load(path);
index.prepare();   // warms rotation matrix, centroids, SIMD-blocked layout
SearchResult result = index.search(query, k);

// Wrong — may return zero scores/ids when loaded from file without prepare()
TurboVecIndex index = TurboVecIndex.load(path);
SearchResult result = index.search(query, k);
```

Verify a loaded index before searching:

```java
Log.d("TurboVec", "dim=" + index.dim() + " size=" + index.size());
// dim > 0 and size > 0 means the index loaded correctly with vectors
```

### Complete Class Reference

#### `SearchResult`

Immutable result object returned by `search(float[], int)`.

| Field / Method | Type | Description |
|---|---|---|
| `scores` | `float[]` | Similarity scores, highest first (row-major for batch queries) |
| `ids` | `long[]` | Stable external IDs matching each score entry |
| `count` | `int` | Number of results filled (`<= queryCount * k`) |
| `isEmpty()` | `boolean` | True when count == 0 |
| `toDebugString(maxRows)` | `String` | Human-readable summary for logging |

#### Constructors & Loaders
- `public TurboVecIndex(int dim, int bitWidth, String filePath) throws IOException`  
  Creates or auto-loads a persisted index. If the file exists, loads it; otherwise creates a fresh one.
- `public static TurboVecIndex create(int dim, int bitWidth)`  
  Creates a new empty in-memory index with no default file path.
- `public static TurboVecIndex load(String filePath) throws IOException`  
  Loads an existing `.tvim` file. **Call `prepare()` before searching.**

#### Mutation (Synchronized)
- `public synchronized void add(float[] vectors, int dim, long[] ids)`  
  Adds `n` vectors (flat array of size `n * dim`) each mapped to a unique external ID.
- `public synchronized boolean remove(long id)`  
  Removes the vector with the given stable ID. Returns `true` if found and removed.
- `public synchronized int removeAll(long[] ids)`  
  Batch-removes IDs; returns count actually removed.
- `public synchronized void save() throws IOException`  
  Saves to the constructor-provided path.
- `public synchronized void save(String filePath) throws IOException`  
  Saves to a specified path.
- `public synchronized void prepare()`  
  Pre-builds search structures (blocked layout, centroid codebook, rotation matrix). **Recommended after every `add()` batch or `load()`.**
- `public synchronized void close()`  
  Frees native memory. Safe with try-with-resources.

#### Search & Query (Lock-Free)
- `public SearchResult search(float[] queries, int k)`  
  Returns a `SearchResult` with `scores`, `ids`, and `count`. Output arrays are allocated internally.
- `public int search(float[] queries, int k, float[] scores, long[] ids)`  
  Zero-allocation variant — writes into caller-supplied arrays, returns filled count. Use in hot paths to avoid GC pressure.
- `public boolean contains(long id)` — true if ID is present in the index.
- `public int size()` — number of indexed vectors.
- `public int dim()` — vector dimensionality (0 if index is uninitialized).
- `public int bitWidth()` — storage bit width (2, 3, 4, or 8).
- `public boolean isClosed()` — true after `close()`.

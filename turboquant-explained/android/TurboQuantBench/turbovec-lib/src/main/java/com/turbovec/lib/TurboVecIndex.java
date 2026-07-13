package com.turbovec.lib;

import java.io.File;
import java.io.IOException;

/**
 * Android wrapper for the TurboVec IdMapIndex — a quantized vector index with stable
 * external long IDs.
 *
 * <p><b>Storage quantization</b> is configured at construction via {@code bitWidth}
 * (2, 3, 4, or 8 bits per coordinate). Vectors are compressed with TurboQuant.</p>
 *
 * <p><b>Stable IDs</b>: unlike the positional indexes, this class assigns each vector a
 * caller-supplied {@code long} ID that remains valid across insertions and deletions.</p>
 *
 * <h3>Thread safety</h3>
 * <ul>
 *   <li>Mutating methods ({@link #add}, {@link #remove}, {@link #removeAll},
 *       {@link #save}, {@link #prepare}, {@link #close}) are {@code synchronized}.</li>
 *   <li>Read methods ({@link #search}, {@link #contains}, {@link #size},
 *       {@link #dim}, {@link #bitWidth}) are <em>not</em> synchronized; the
 *       underlying Rust search takes {@code &self} and is safe for concurrent
 *       readers when no writer holds the lock.</li>
 * </ul>
 *
 * <h3>Lifecycle</h3>
 * <p>Always call {@link #close()} or use try-with-resources to free native memory.</p>
 */
public final class TurboVecIndex implements AutoCloseable {

    static {
        System.loadLibrary("tqbench");
    }

    private volatile long handle;

    /** File path used by {@link #save()} with no arguments. May be null. */
    private final String defaultFilePath;

    // ─── Constructors ───────────────────────────────────────────────────────────

    /**
     * Creates a new index, or loads an existing one from {@code filePath} if that file
     * already exists.
     *
     * <p>This is the primary constructor: pass the path where the index should be
     * persisted. On first run the file does not exist and a fresh index is created.
     * On subsequent runs (or after {@link #save()}), the file exists and is loaded
     * automatically.</p>
     *
     * @param dim      vector dimensionality; must be positive and a multiple of 8
     * @param bitWidth quantization bit width — one of 2, 3, 4, or 8
     * @param filePath path to load from / save to (must not be null or empty)
     * @throws IOException              if the file exists but cannot be read
     * @throws IllegalArgumentException if dim or bitWidth is invalid
     */
    public TurboVecIndex(int dim, int bitWidth, String filePath) throws IOException {
        if (filePath == null || filePath.isEmpty()) {
            throw new IllegalArgumentException("filePath must not be null or empty");
        }
        this.defaultFilePath = filePath;
        if (new File(filePath).exists()) {
            this.handle = nativeIdMapLoad(filePath);
            if (this.handle == 0L) {
                throw new IOException("Failed to load index from: " + filePath);
            }
        } else {
            validateDim(dim);
            validateBitWidth(bitWidth);
            this.handle = nativeIdMapNew(dim, bitWidth);
            if (this.handle == 0L) {
                throw new IllegalArgumentException(
                        "Failed to create index: dim=" + dim + ", bitWidth=" + bitWidth);
            }
        }
    }

    /** Private constructor used by {@link #create} and {@link #load}. */
    private TurboVecIndex(long handle, String defaultFilePath) {
        if (handle == 0L) {
            throw new IllegalStateException("native turbovec handle is null");
        }
        this.handle = handle;
        this.defaultFilePath = defaultFilePath;
    }

    /**
     * Creates a new empty index without a default file path.
     *
     * @param dim      vector dimensionality; must be positive and a multiple of 8
     * @param bitWidth quantization bit width — one of 2, 3, 4, or 8
     */
    public static TurboVecIndex create(int dim, int bitWidth) {
        validateDim(dim);
        validateBitWidth(bitWidth);
        return new TurboVecIndex(nativeIdMapNew(dim, bitWidth), null);
    }

    /**
     * Loads an existing index from a {@code .tvim} file written by {@link #save(String)}.
     *
     * @param filePath path to the {@code .tvim} file
     * @throws IOException if the file cannot be read
     */
    public static TurboVecIndex load(String filePath) throws IOException {
        if (filePath == null || filePath.isEmpty()) {
            throw new IllegalArgumentException("filePath must not be null or empty");
        }
        long h = nativeIdMapLoad(filePath);
        if (h == 0L) {
            throw new IOException("Failed to load index from: " + filePath);
        }
        return new TurboVecIndex(h, filePath);
    }

    // ─── Write operations (synchronized) ────────────────────────────────────────

    /**
     * Adds {@code n} vectors with their corresponding stable external IDs.
     *
     * <p>The vectors array must be flat row-major with length {@code n * dim}.
     * The ids array must have exactly {@code n} elements. Each ID must be unique
     * and not already present in the index.</p>
     *
     * @param vectors flat float array of length {@code n * dim}
     * @param dim     dimensionality of each vector (must match index dim)
     * @param ids     array of {@code n} unique external long IDs
     * @throws IllegalArgumentException if the array lengths are inconsistent,
     *                                  any ID is already present, or dim mismatches
     */
    public synchronized void add(float[] vectors, int dim, int[] ids) {
        ensureOpen();
        if (vectors == null) throw new IllegalArgumentException("vectors must not be null");
        if (ids == null) throw new IllegalArgumentException("ids must not be null");
        if (dim <= 0) throw new IllegalArgumentException("dim must be positive");
        if (vectors.length % dim != 0) {
            throw new IllegalArgumentException(
                    "vectors.length (" + vectors.length + ") must be a multiple of dim (" + dim + ")");
        }
        int n = vectors.length / dim;
        if (ids.length != n) {
            throw new IllegalArgumentException(
                    "ids.length (" + ids.length + ") must equal vector count (" + n + ")");
        }
        nativeIdMapAdd(handle, vectors, dim, ids);
    }

    /**
     * Removes the vector with the given external ID.
     *
     * @param id external ID to remove
     * @return {@code true} if the ID was present and removed; {@code false} otherwise
     */
    public synchronized boolean remove(int id) {
        ensureOpen();
        return nativeIdMapRemove(handle, id);
    }

    /**
     * Batch-removes multiple external IDs. Silently skips IDs that are not present.
     *
     * @param ids array of external IDs to remove
     * @return number of IDs actually removed
     */
    public synchronized int removeAll(int[] ids) {
        ensureOpen();
        if (ids == null || ids.length == 0) return 0;
        int removed = 0;
        for (long id : ids) {
            if (nativeIdMapRemove(handle, id)) removed++;
        }
        return removed;
    }

    /**
     * Saves the index to the file path supplied at construction.
     *
     * @throws IOException              if writing fails
     * @throws IllegalStateException    if no default file path was set (use {@link #save(String)})
     */
    public synchronized void save() throws IOException {
        ensureOpen();
        if (defaultFilePath == null) {
            throw new IllegalStateException(
                    "No default file path. Use save(String filePath) instead.");
        }
        nativeIdMapWrite(handle, defaultFilePath);
    }

    /**
     * Saves the index to the specified file path.
     *
     * @param filePath path to write the {@code .tvim} file
     * @throws IOException if writing fails
     */
    public synchronized void save(String filePath) throws IOException {
        ensureOpen();
        if (filePath == null || filePath.isEmpty()) {
            throw new IllegalArgumentException("filePath must not be null or empty");
        }
        nativeIdMapWrite(handle, filePath);
    }

    /**
     * Warms the native search caches (rotation matrix, centroids, SIMD-blocked layout).
     * Optional but recommended after bulk {@link #add} to avoid cold first-query latency.
     */
    public synchronized void prepare() {
        ensureOpen();
        nativeIdMapPrepare(handle);
    }

    /**
     * Releases native memory. After this call all methods throw {@link IllegalStateException}.
     * Safe to call multiple times.
     */
    @Override
    public synchronized void close() {
        if (handle != 0L) {
            nativeIdMapFree(handle);
            handle = 0L;
        }
    }

    // ─── Read operations (not synchronized) ─────────────────────────────────────

    /**
     * Searches for the top-{@code k} nearest vectors for each query and writes results
     * into the caller-supplied output arrays.
     *
     * <p>Output arrays must be pre-allocated to size {@code queryCount * k}:
     * <ul>
     *   <li>{@code scores[qi*k .. (qi+1)*k]} — similarity scores for query {@code qi}</li>
     *   <li>{@code ids[qi*k .. (qi+1)*k]} — external IDs of matched vectors for query {@code qi}</li>
     * </ul>
     *
     * @param queries flat float array of {@code queryCount * dim} values
     * @param k       maximum results per query
     * @param scores  output: pre-allocated {@code float[queryCount * k]}
     * @param ids     output: pre-allocated {@code long[queryCount * k]}
     * @return actual number of results filled ({@code ≤ queryCount * k}); may be
     *         less when the index has fewer than {@code k} vectors
     * @throws IllegalArgumentException if queries is null, k is negative,
     *                                  or output arrays are too small
     */
    public int search(float[] queries, int k, float[] scores, int[] ids) {
        long h = handle;
        if (h == 0L) throw new IllegalStateException("TurboVecIndex is closed");
        if (queries == null) throw new IllegalArgumentException("queries must not be null");
        if (k < 0) throw new IllegalArgumentException("k must be non-negative");
        if (scores == null || ids == null) {
            throw new IllegalArgumentException("scores and ids output arrays must not be null");
        }
        int currentDim = nativeIdMapDim(h);
        if (currentDim > 0 && queries.length % currentDim != 0) {
            throw new IllegalArgumentException(
                    "queries.length must be a multiple of index dim (" + currentDim + ")");
        }
        int nq = currentDim > 0 ? queries.length / currentDim : 0;
        int required = nq * k;
        if (scores.length < required) {
            throw new IllegalArgumentException(
                    "scores array too small: need " + required + ", got " + scores.length);
        }
        if (ids.length < required) {
            throw new IllegalArgumentException(
                    "ids array too small: need " + required + ", got " + ids.length);
        }
        return nativeIdMapSearch(h, queries, k, scores, ids);
    }

    /**
     * Searches for the top-{@code k} nearest vectors and returns results as a
     * {@link SearchResult} object.
     *
     * <p>This is a convenience overload that allocates the output arrays internally.
     * For hot paths where you want to reuse pre-allocated buffers, use
     * {@link #search(float[], int, float[], long[])} instead.</p>
     *
     * @param queries flat float array of {@code queryCount * dim} values (single or batch)
     * @param k       maximum results per query
     * @return a {@link SearchResult} containing {@code scores}, {@code ids}, and {@code count};
     *         never {@code null} — returns an empty result when the index is empty or {@code k == 0}
     * @throws IllegalArgumentException if queries is null or k is negative
     * @throws IllegalStateException    if the index has been closed
     */
    public SearchResult search(float[] queries, int k) {
        long h = handle;
        if (h == 0L) throw new IllegalStateException("TurboVecIndex is closed");
        if (queries == null) throw new IllegalArgumentException("queries must not be null");
        if (k < 0)  throw new IllegalArgumentException("k must be non-negative");

        int currentDim = nativeIdMapDim(h);
        if (currentDim > 0 && queries.length % currentDim != 0) {
            throw new IllegalArgumentException(
                    "queries.length must be a multiple of index dim (" + currentDim + ")");
        }
        int nq = currentDim > 0 ? queries.length / currentDim : 0;
        if (nq == 0 || k == 0) {
            return new SearchResult(new float[0], new long[0], 0);
        }

        // Delegate entirely to native: it allocates float[], long[], and the
        // SearchResult object itself via JNI — no Java-side allocation here.
        SearchResult result = nativeIdMapSearchResult(h, queries, k);
        return result != null ? result : new SearchResult(new float[0], new long[0], 0);
    }

    /**
     * Returns {@code true} if the index currently contains a vector with this ID.
     */
    public boolean contains(int id) {
        long h = handle;
        if (h == 0L) throw new IllegalStateException("TurboVecIndex is closed");
        return nativeIdMapContains(h, id);
    }

    /**
     * Returns the number of vectors currently in the index.
     */
    public int size() {
        long h = handle;
        if (h == 0L) throw new IllegalStateException("TurboVecIndex is closed");
        return nativeIdMapSize(h);
    }

    /**
     * Returns the committed vector dimensionality, or 0 for an uncommitted lazy index.
     */
    public int dim() {
        long h = handle;
        if (h == 0L) throw new IllegalStateException("TurboVecIndex is closed");
        return nativeIdMapDim(h);
    }

    /**
     * Returns the TurboQuant storage bit width (2, 3, 4, or 8).
     */
    public int bitWidth() {
        long h = handle;
        if (h == 0L) throw new IllegalStateException("TurboVecIndex is closed");
        return nativeIdMapBitWidth(h);
    }

    /**
     * Returns {@code true} if this index has been closed and native memory freed.
     */
    public boolean isClosed() {
        return handle == 0L;
    }

    // ─── Internal ────────────────────────────────────────────────────────────────

    private void ensureOpen() {
        if (handle == 0L) throw new IllegalStateException("TurboVecIndex is closed");
    }

    private static void validateDim(int dim) {
        if (dim <= 0 || dim % 8 != 0) {
            throw new IllegalArgumentException(
                    "dim must be positive and a multiple of 8, got: " + dim);
        }
    }

    private static void validateBitWidth(int bitWidth) {
        if (bitWidth != 2 && bitWidth != 3 && bitWidth != 4 && bitWidth != 8) {
            throw new IllegalArgumentException(
                    "bitWidth must be one of 2, 3, 4, or 8, got: " + bitWidth);
        }
    }

    // ─── Native declarations ──────────────────────────────────────────────────────

    private static native long        nativeIdMapNew(int dim, int bitWidth);
    private static native long        nativeIdMapLoad(String path) throws IOException;
    private static native void        nativeIdMapFree(long handle);
    private static native void        nativeIdMapAdd(long handle, float[] vectors, int dim, int[] ids);
    private static native int         nativeIdMapSearch(long handle, float[] queries, int k,
                                                        float[] outScores, int[] outIds);
    private static native SearchResult nativeIdMapSearchResult(long handle, float[] queries, int k);
    private static native boolean     nativeIdMapRemove(long handle, long id);
    private static native void        nativeIdMapWrite(long handle, String path) throws IOException;
    private static native void        nativeIdMapPrepare(long handle);
    private static native int         nativeIdMapSize(long handle);
    private static native int         nativeIdMapDim(long handle);
    private static native int         nativeIdMapBitWidth(long handle);
    private static native boolean     nativeIdMapContains(long handle, long id);
}

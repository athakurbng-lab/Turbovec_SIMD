package com.turbovec.lib;

/**
 * Immutable result object returned by {@link TurboVecIndex#search(float[], int)}.
 *
 * <p>Results are stored in row-major order. For a single-query search the arrays
 * are simply {@code scores[0..count-1]} and {@code ids[0..count-1]}.
 * For a multi-query batch (e.g. {@code queryCount} queries each requesting top-{@code k}),
 * row {@code qi} occupies {@code scores[qi*k .. (qi+1)*k-1]} and
 * {@code ids[qi*k .. (qi+1)*k-1]}.</p>
 *
 * <p>The actual number of populated entries is {@link #count}. This may be less than
 * {@code queryCount * k} when the index contains fewer than {@code k} vectors.</p>
 */
public final class SearchResult {

    /** Similarity scores, highest first within each query row. */
    public final float[] scores;

    /** Stable external int IDs corresponding to each score entry. */
    public final int[] ids;

    /**
     * Total number of results filled across all queries.
     * Equals {@code nq * actualK} where {@code actualK = min(k, index.size())}.
     */
    public final int count;

    /** Package-private constructor — created only by {@link TurboVecIndex}. */
    SearchResult(float[] scores, int[] ids, int count) {
        this.scores = scores;
        this.ids    = ids;
        this.count  = count;
    }

    /**
     * Convenience: true when no results were returned (empty index or k == 0).
     */
    public boolean isEmpty() {
        return count == 0;
    }

    /**
     * Returns a human-readable summary of the top results, useful for logging.
     *
     * @param maxRows maximum number of result rows to include
     */
    public String toDebugString(int maxRows) {
        int show = Math.min(count, maxRows);
        StringBuilder sb = new StringBuilder("SearchResult{count=").append(count).append(", top=[");
        for (int i = 0; i < show; i++) {
            if (i > 0) sb.append(", ");
            sb.append("(id=").append(ids[i])
              .append(", score=").append(String.format("%.4f", scores[i]))
              .append(')');
        }
        if (show < count) sb.append(", ...");
        sb.append("]}");
        return sb.toString();
    }

    @Override
    public String toString() {
        return toDebugString(5);
    }
}

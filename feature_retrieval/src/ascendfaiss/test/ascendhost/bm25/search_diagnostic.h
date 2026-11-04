#ifndef BM25_TEST_SEARCH_DIAGNOSTIC_H
#define BM25_TEST_SEARCH_DIAGNOSTIC_H
#include "ascendhost/src/bm25/large_search.h"
namespace bm25large {
// Test-only access: destructive checks must never run inside measured Search calls.
class SessionDiagnostics {
public:
    static void Poison(Session &session) { session.PoisonForDiagnostic(); }
    static void VerifyIndex(Session &session) { session.VerifyIndex(); }
    static void Save(Session &session, const Queries &queries, const PlannedResult &result, const fs::path &path,
                     const fs::path &reference)
    {
        session.SaveDiagnostic(queries, result, path, reference);
    }
};
} // namespace bm25large
#endif // BM25_TEST_SEARCH_DIAGNOSTIC_H

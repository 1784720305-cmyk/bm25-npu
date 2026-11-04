#ifndef BM25_TEST_DLFCN_H
#define BM25_TEST_DLFCN_H
// Header-only API declarations let lifecycle tests compile on Windows.
constexpr int RTLD_NOW = 0;
constexpr int RTLD_LOCAL = 0;
struct Dl_info {
    const char *dli_fname;
};
#ifdef _WIN32
void *dlopen(const wchar_t *, int);
#else
void *dlopen(const char *, int);
#endif
const char *dlerror();
void *dlsym(void *, const char *);
int dladdr(void *, Dl_info *);
#endif // BM25_TEST_DLFCN_H

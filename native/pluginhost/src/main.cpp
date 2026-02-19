#include <clap/clap.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
  #include <windows.h>
#endif

namespace fs = std::filesystem;

static std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned int)(unsigned char)c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

static std::vector<std::string> split_paths(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
#if defined(_WIN32)
    const char sep = ';';
#else
    const char sep = ':';
#endif
    if (c == sep) {
      if (!cur.empty()) out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

static std::optional<std::string> getenv_str(const char* key) {
  const char* v = std::getenv(key);
  if (!v) return std::nullopt;
  std::string s(v);
  // trim
  while (!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
  while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
  if (s.empty()) return std::nullopt;
  return s;
}

static bool ieq_ext(const fs::path& p, const std::string& ext) {
  auto e = p.extension().string();
  if (e.size() != ext.size()) return false;
  for (size_t i = 0; i < e.size(); i++) {
    if (std::tolower((unsigned char)e[i]) != std::tolower((unsigned char)ext[i])) return false;
  }
  return true;
}

static std::vector<fs::path> default_clap_paths() {
  std::vector<fs::path> out;
#if defined(_WIN32)
  char* common = nullptr;
  size_t len = 0;
  if (_dupenv_s(&common, &len, "COMMONPROGRAMFILES") == 0 && common) {
    out.emplace_back(fs::path(common) / "CLAP");
    std::free(common);
  }
  char* local = nullptr;
  len = 0;
  if (_dupenv_s(&local, &len, "LOCALAPPDATA") == 0 && local) {
    out.emplace_back(fs::path(local) / "CLAP");
    std::free(local);
  }
#else
  out.emplace_back("/usr/lib/clap");
  out.emplace_back("/usr/local/lib/clap");
#endif

  if (auto extra = getenv_str("SOUNDGEN_CLAP_PATHS")) {
    for (const auto& s : split_paths(*extra)) out.emplace_back(s);
  }
  return out;
}

static std::vector<fs::path> default_lv2_paths() {
  std::vector<fs::path> out;
#if defined(_WIN32)
  char* appdata = nullptr;
  size_t len = 0;
  if (_dupenv_s(&appdata, &len, "APPDATA") == 0 && appdata) {
    out.emplace_back(fs::path(appdata) / "LV2");
    std::free(appdata);
  }
  char* common = nullptr;
  len = 0;
  if (_dupenv_s(&common, &len, "COMMONPROGRAMFILES") == 0 && common) {
    out.emplace_back(fs::path(common) / "LV2");
    std::free(common);
  }
#else
  out.emplace_back("/usr/lib/lv2");
  out.emplace_back("/usr/local/lib/lv2");
#endif

  if (auto env = getenv_str("LV2_PATH")) {
    for (const auto& s : split_paths(*env)) out.emplace_back(s);
  }
  return out;
}

static std::vector<fs::path> scan_clap() {
  std::vector<fs::path> found;
  for (const auto& base : default_clap_paths()) {
    std::error_code ec;
    if (!fs::exists(base, ec)) continue;
    for (auto it = fs::recursive_directory_iterator(base, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
      if (ec) break;
      const auto& p = it->path();
      if (!it->is_regular_file(ec)) continue;
      if (ieq_ext(p, ".clap")) found.push_back(p);
    }
  }
  std::sort(found.begin(), found.end());
  found.erase(std::unique(found.begin(), found.end()), found.end());
  return found;
}

static std::vector<fs::path> scan_lv2() {
  std::vector<fs::path> found;
  for (const auto& base : default_lv2_paths()) {
    std::error_code ec;
    if (!fs::exists(base, ec)) continue;
    for (auto it = fs::directory_iterator(base, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
      if (ec) break;
      const auto& p = it->path();
      if (!it->is_directory(ec)) continue;
      if (ieq_ext(p, ".lv2")) found.push_back(p);
    }
  }
  std::sort(found.begin(), found.end());
  found.erase(std::unique(found.begin(), found.end()), found.end());
  return found;
}

#pragma pack(push, 1)
struct WavHeader {
  char riff[4];
  uint32_t riffSize;
  char wave[4];
  char fmt[4];
  uint32_t fmtSize;
  uint16_t audioFormat;
  uint16_t numChannels;
  uint32_t sampleRate;
  uint32_t byteRate;
  uint16_t blockAlign;
  uint16_t bitsPerSample;
};
#pragma pack(pop)

struct WavData {
  uint32_t sampleRate = 44100;
  uint16_t channels = 1;
  std::vector<float> samples; // interleaved
};

static bool read_wav_pcm16(const fs::path& path, WavData& out, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { err = "failed to open input wav"; return false; }

  WavHeader h{};
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (!f) { err = "invalid wav header"; return false; }
  if (std::string_view(h.riff, 4) != "RIFF" || std::string_view(h.wave, 4) != "WAVE") {
    err = "not a RIFF/WAVE file";
    return false;
  }

  // Walk chunks to find 'fmt ' and 'data'.
  bool haveFmt = false;
  bool haveData = false;
  uint16_t fmtChannels = 0;
  uint32_t fmtRate = 0;
  uint16_t fmtBits = 0;
  uint16_t fmtFormat = 0;

  // We already read a struct that assumes fmt immediately after RIFF, but real WAV may differ.
  // Rewind and parse properly.
  f.clear();
  f.seekg(12, std::ios::beg);

  while (f) {
    char id[4];
    uint32_t sz = 0;
    f.read(id, 4);
    if (!f) break;
    f.read(reinterpret_cast<char*>(&sz), 4);
    if (!f) break;

    const std::string_view sid(id, 4);
    if (sid == "fmt ") {
      std::vector<char> buf(sz);
      f.read(buf.data(), sz);
      if (!f) { err = "truncated fmt chunk"; return false; }
      if (sz < 16) { err = "fmt chunk too small"; return false; }
      fmtFormat = *reinterpret_cast<uint16_t*>(&buf[0]);
      fmtChannels = *reinterpret_cast<uint16_t*>(&buf[2]);
      fmtRate = *reinterpret_cast<uint32_t*>(&buf[4]);
      fmtBits = *reinterpret_cast<uint16_t*>(&buf[14]);
      haveFmt = true;
    } else if (sid == "data") {
      if (!haveFmt) { err = "data chunk before fmt"; return false; }
      if (fmtFormat != 1 || fmtBits != 16) {
        err = "only PCM16 supported";
        return false;
      }
      const size_t nSamp = sz / sizeof(int16_t);
      std::vector<int16_t> pcm(nSamp);
      f.read(reinterpret_cast<char*>(pcm.data()), sz);
      if (!f) { err = "truncated data chunk"; return false; }

      out.sampleRate = fmtRate;
      out.channels = fmtChannels;
      out.samples.resize(nSamp);
      for (size_t i = 0; i < nSamp; i++) {
        out.samples[i] = std::clamp((float)pcm[i] / 32768.0f, -1.0f, 1.0f);
      }
      haveData = true;
      break;
    } else {
      // skip unknown chunk
      f.seekg(sz, std::ios::cur);
    }

    // align to even
    if (sz & 1) f.seekg(1, std::ios::cur);
  }

  if (!haveData) { err = "no data chunk"; return false; }
  return true;
}

static bool write_wav_pcm16(const fs::path& path, const WavData& in, std::string& err) {
  if (in.channels == 0) { err = "invalid channels"; return false; }

  const uint32_t dataBytes = (uint32_t)(in.samples.size() * sizeof(int16_t));
  const uint32_t fmtSize = 16;
  const uint32_t riffSize = 4 + (8 + fmtSize) + (8 + dataBytes);

  std::ofstream f(path, std::ios::binary);
  if (!f) { err = "failed to open output wav"; return false; }

  f.write("RIFF", 4);
  f.write(reinterpret_cast<const char*>(&riffSize), 4);
  f.write("WAVE", 4);

  f.write("fmt ", 4);
  f.write(reinterpret_cast<const char*>(&fmtSize), 4);
  uint16_t audioFormat = 1;
  uint16_t numChannels = in.channels;
  uint32_t sampleRate = in.sampleRate;
  uint16_t bitsPerSample = 16;
  uint16_t blockAlign = (uint16_t)(numChannels * (bitsPerSample / 8));
  uint32_t byteRate = sampleRate * blockAlign;

  f.write(reinterpret_cast<const char*>(&audioFormat), 2);
  f.write(reinterpret_cast<const char*>(&numChannels), 2);
  f.write(reinterpret_cast<const char*>(&sampleRate), 4);
  f.write(reinterpret_cast<const char*>(&byteRate), 4);
  f.write(reinterpret_cast<const char*>(&blockAlign), 2);
  f.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

  f.write("data", 4);
  f.write(reinterpret_cast<const char*>(&dataBytes), 4);

  for (float v : in.samples) {
    const float c = std::clamp(v, -1.0f, 1.0f);
    int16_t s = (int16_t)std::lrintf(c * 32767.0f);
    f.write(reinterpret_cast<const char*>(&s), 2);
  }

  return true;
}

struct ClapLibrary {
#if defined(_WIN32)
  HMODULE mod = nullptr;
#else
  void* mod = nullptr;
#endif
  const clap_plugin_entry_t* entry = nullptr;

  bool load(const fs::path& p, std::string& err) {
#if defined(_WIN32)
    mod = LoadLibraryW(p.wstring().c_str());
    if (!mod) { err = "LoadLibrary failed"; return false; }
    auto sym = (const clap_plugin_entry_t*(*)())GetProcAddress(mod, "clap_entry");
    if (!sym) {
      err = "missing clap_entry";
      return false;
    }
    entry = sym();
#else
    err = "non-windows not implemented";
    return false;
#endif

    if (!entry) { err = "null clap entry"; return false; }
    if (!entry->init("soundgen_pluginhost")) {
      err = "clap entry init failed";
      return false;
    }
    return true;
  }

  void unload() {
    if (entry) {
      entry->deinit();
      entry = nullptr;
    }
#if defined(_WIN32)
    if (mod) {
      FreeLibrary(mod);
      mod = nullptr;
    }
#endif
  }

  ~ClapLibrary() { unload(); }
};

static const clap_plugin_factory_t* get_factory(const ClapLibrary& lib) {
  if (!lib.entry) return nullptr;
  return (const clap_plugin_factory_t*)lib.entry->get_factory(CLAP_PLUGIN_FACTORY_ID);
}

static int clap_plugin_count(const ClapLibrary& lib) {
  const auto* f = get_factory(lib);
  if (!f) return 0;
  return (int)f->get_plugin_count(f);
}

static const clap_plugin_descriptor_t* clap_get_desc(const ClapLibrary& lib, int idx) {
  const auto* f = get_factory(lib);
  if (!f) return nullptr;
  return f->get_plugin_descriptor(f, (uint32_t)idx);
}

static int cmd_scan() {
  const auto claps = scan_clap();
  const auto lv2s = scan_lv2();

  std::cout << "{";
  std::cout << "\"clap\":[";
  for (size_t i = 0; i < claps.size(); i++) {
    if (i) std::cout << ",";
    std::cout << "\"" << json_escape(claps[i].string()) << "\"";
  }
  std::cout << "],\"lv2\":[";
  for (size_t i = 0; i < lv2s.size(); i++) {
    if (i) std::cout << ",";
    std::cout << "\"" << json_escape(lv2s[i].string()) << "\"";
  }
  std::cout << "]}";
  std::cout << std::endl;
  return 0;
}

static int cmd_clap_list(const fs::path& pluginPath) {
  std::string err;
  ClapLibrary lib;
  if (!lib.load(pluginPath, err)) {
    std::cout << "{\"ok\":false,\"error\":\"" << json_escape(err) << "\"}" << std::endl;
    return 2;
  }
  const int n = clap_plugin_count(lib);
  std::cout << "{\"ok\":true,\"count\":" << n << ",\"plugins\":[";
  for (int i = 0; i < n; i++) {
    const auto* d = clap_get_desc(lib, i);
    if (!d) continue;
    if (i) std::cout << ",";
    std::cout << "{";
    std::cout << "\"id\":\"" << json_escape(d->id ? d->id : "") << "\",";
    std::cout << "\"name\":\"" << json_escape(d->name ? d->name : "") << "\"";
    std::cout << "}";
  }
  std::cout << "]}" << std::endl;
  return 0;
}

// NOTE: this is intentionally limited; it aims to support common simple FX.
static int cmd_clap_render(const fs::path& pluginPath, const fs::path& inWav, const fs::path& outWav) {
  std::string err;
  WavData wav;
  if (!read_wav_pcm16(inWav, wav, err)) {
    std::cout << "{\"ok\":false,\"error\":\"" << json_escape("wav read: " + err) << "\"}" << std::endl;
    return 2;
  }

  // Downmix to mono for now.
  if (wav.channels > 1) {
    const size_t frames = wav.samples.size() / wav.channels;
    std::vector<float> mono(frames);
    for (size_t i = 0; i < frames; i++) {
      float acc = 0.0f;
      for (size_t ch = 0; ch < wav.channels; ch++) acc += wav.samples[i * wav.channels + ch];
      mono[i] = acc / (float)wav.channels;
    }
    wav.channels = 1;
    wav.samples.swap(mono);
  }

  ClapLibrary lib;
  if (!lib.load(pluginPath, err)) {
    std::cout << "{\"ok\":false,\"error\":\"" << json_escape(err) << "\"}" << std::endl;
    return 2;
  }

  const auto* factory = get_factory(lib);
  if (!factory) {
    std::cout << "{\"ok\":false,\"error\":\"missing plugin factory\"}" << std::endl;
    return 2;
  }

  const int n = clap_plugin_count(lib);
  if (n <= 0) {
    std::cout << "{\"ok\":false,\"error\":\"no plugins in library\"}" << std::endl;
    return 2;
  }

  // Use the first plugin in the library.
  const auto* desc = clap_get_desc(lib, 0);
  if (!desc || !desc->id) {
    std::cout << "{\"ok\":false,\"error\":\"invalid plugin descriptor\"}" << std::endl;
    return 2;
  }

  // Minimal host.
  clap_host_t host{};
  host.clap_version = CLAP_VERSION;
  host.name = "SÖNDBÖUND";
  host.vendor = "soundgen";
  host.url = "";
  host.version = "0";
  host.get_extension = [](const clap_host_t*, const char*) -> const void* { return nullptr; };
  host.request_restart = [](const clap_host_t*) {};
  host.request_process = [](const clap_host_t*) {};
  host.request_callback = [](const clap_host_t*) {};

  const clap_plugin_t* plugin = factory->create_plugin(factory, &host, desc->id);
  if (!plugin) {
    std::cout << "{\"ok\":false,\"error\":\"create_plugin failed\"}" << std::endl;
    return 2;
  }

  if (!plugin->init(plugin)) {
    plugin->destroy(plugin);
    std::cout << "{\"ok\":false,\"error\":\"plugin init failed\"}" << std::endl;
    return 2;
  }

  const double sr = (double)wav.sampleRate;
  const uint32_t minFrames = 1;
  const uint32_t maxFrames = 4096;
  if (!plugin->activate(plugin, sr, minFrames, maxFrames)) {
    plugin->destroy(plugin);
    std::cout << "{\"ok\":false,\"error\":\"activate failed\"}" << std::endl;
    return 2;
  }

  if (plugin->start_processing) {
    if (!plugin->start_processing(plugin)) {
      plugin->deactivate(plugin);
      plugin->destroy(plugin);
      std::cout << "{\"ok\":false,\"error\":\"start_processing failed\"}" << std::endl;
      return 2;
    }
  }

  const size_t frames = wav.samples.size(); // mono
  std::vector<float> out(frames);

  const uint32_t block = 1024;
  std::vector<float> inBlock(block);
  std::vector<float> outBlock(block);

  // Set up audio buffers for process.
  clap_audio_buffer_t inBuf{};
  clap_audio_buffer_t outBuf{};

  float* inChans[1] = { inBlock.data() };
  float* outChans[1] = { outBlock.data() };

  inBuf.channel_count = 1;
  inBuf.data32 = (float**)inChans;
  inBuf.data64 = nullptr;
  outBuf.channel_count = 1;
  outBuf.data32 = (float**)outChans;
  outBuf.data64 = nullptr;

  clap_process_t pr{};
  pr.steady_time = 0;
  pr.frames_count = 0;
  pr.transport = nullptr;
  pr.audio_inputs = &inBuf;
  pr.audio_inputs_count = 1;
  pr.audio_outputs = &outBuf;
  pr.audio_outputs_count = 1;
  pr.in_events = nullptr;
  pr.out_events = nullptr;

  size_t pos = 0;
  while (pos < frames) {
    const uint32_t nfrm = (uint32_t)std::min<size_t>(block, frames - pos);
    std::copy(wav.samples.begin() + pos, wav.samples.begin() + pos + nfrm, inBlock.begin());
    if (nfrm < block) std::fill(inBlock.begin() + nfrm, inBlock.end(), 0.0f);

    pr.frames_count = nfrm;
    const auto st = plugin->process(plugin, &pr);
    if (st == CLAP_PROCESS_ERROR) {
      if (plugin->stop_processing) plugin->stop_processing(plugin);
      plugin->deactivate(plugin);
      plugin->destroy(plugin);
      std::cout << "{\"ok\":false,\"error\":\"process error\"}" << std::endl;
      return 2;
    }

    std::copy(outBlock.begin(), outBlock.begin() + nfrm, out.begin() + pos);
    pos += nfrm;
  }

  if (plugin->stop_processing) plugin->stop_processing(plugin);
  plugin->deactivate(plugin);
  plugin->destroy(plugin);

  WavData outW{};
  outW.sampleRate = wav.sampleRate;
  outW.channels = 1;
  outW.samples = std::move(out);

  if (!write_wav_pcm16(outWav, outW, err)) {
    std::cout << "{\"ok\":false,\"error\":\"" << json_escape("wav write: " + err) << "\"}" << std::endl;
    return 2;
  }

  std::cout << "{\"ok\":true,\"plugin_id\":\"" << json_escape(desc->id) << "\"}" << std::endl;
  return 0;
}

static void usage() {
  std::cerr << "soundgen_pluginhost commands:\n";
  std::cerr << "  scan\n";
  std::cerr << "  clap-list --plugin <path>\n";
  std::cerr << "  clap-render --plugin <path> --in <wav> --out <wav>\n";
}

static std::optional<std::string> arg_value(int& i, int argc, char** argv) {
  if (i + 1 >= argc) return std::nullopt;
  i++;
  return std::string(argv[i]);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }

  const std::string cmd(argv[1]);
  if (cmd == "scan") {
    return cmd_scan();
  }

  if (cmd == "clap-list") {
    std::optional<std::string> plugin;
    for (int i = 2; i < argc; i++) {
      std::string a(argv[i]);
      if (a == "--plugin") plugin = arg_value(i, argc, argv);
    }
    if (!plugin) {
      usage();
      return 2;
    }
    return cmd_clap_list(fs::path(*plugin));
  }

  if (cmd == "clap-render") {
    std::optional<std::string> plugin;
    std::optional<std::string> in;
    std::optional<std::string> out;
    for (int i = 2; i < argc; i++) {
      std::string a(argv[i]);
      if (a == "--plugin") plugin = arg_value(i, argc, argv);
      else if (a == "--in") in = arg_value(i, argc, argv);
      else if (a == "--out") out = arg_value(i, argc, argv);
    }
    if (!plugin || !in || !out) {
      usage();
      return 2;
    }
    return cmd_clap_render(fs::path(*plugin), fs::path(*in), fs::path(*out));
  }

  usage();
  return 2;
}

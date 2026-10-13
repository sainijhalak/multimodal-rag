'use client';

import React, { useState, useEffect } from 'react';
import { 
  Sparkles, CloudArrowUp, MessageSquare, FolderOpen, 
  ArrowUp, User, Bot, RefreshCw, ExternalLink 
} from 'lucide-react';

interface SourceItem {
  title: string;
  content: string;
  source: string;
  modality: string;
  similarity: number;
}

interface UploadedFileItem {
  name: string;
  extension: string;
  modality: string;
  size: string;
  url: string;
}

export default function Home() {
  const [apiUrl, setApiUrl] = useState<string>(
    process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  );
  const [activeTab, setActiveTab] = useState<'chat' | 'upload'>('chat');
  const [gpuStatus, setGpuStatus] = useState<string>('RTX 3050 Ti • Local');
  const [query, setQuery] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [displayedQuery, setDisplayedQuery] = useState<string | null>(null);
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [fileList, setFileList] = useState<UploadedFileItem[]>([]);

  const fetchHealthAndFiles = async () => {
    try {
      const res = await fetch(`${apiUrl}/health/`);
      if (res.ok) {
        const data = await res.json();
        const dev = data.device ? data.device.replace('NVIDIA GeForce ', '').replace(' Laptop GPU', '') : 'CUDA';
        setGpuStatus(`${dev} • ${data.total_documents_indexed || 0} items`);
      }
    } catch (e) {}

    try {
      const resFiles = await fetch(`${apiUrl}/ingest/list`);
      if (resFiles.ok) {
        const data = await resFiles.json();
        setFileList(data.files || []);
      }
    } catch (e) {}
  };

  useEffect(() => {
    fetchHealthAndFiles();
  }, [apiUrl]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const q = query.trim();
    setDisplayedQuery(q);
    setQuery('');
    setLoading(true);
    setAiAnswer(null);
    setSources([]);
    setActiveTab('chat');
    const start = performance.now();

    try {
      const res = await fetch(`${apiUrl}/query/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: 4 }),
      });
      const data = await res.json();
      setAiAnswer(data.answer || 'No direct answer returned.');
      setSources(data.sources || []);
    } catch (err: any) {
      setAiAnswer('Request error: ' + err.message);
    } finally {
      setElapsedMs(Math.round(performance.now() - start));
      setLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setUploadStatus(`Indexing ${file.name} on GPU...`);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${apiUrl}/ingest/file`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`✓ Successfully indexed ${data.filename}!`);
        fetchHealthAndFiles();
        setTimeout(() => setUploadStatus(null), 5000);
      } else {
        setUploadStatus(`Error: ${data.detail || 'Upload failed'}`);
      }
    } catch (err: any) {
      setUploadStatus(`Upload failed: ${err.message}`);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* Top Navbar */}
      <header className="border-b border-slate-900 bg-slate-950/90 backdrop-blur sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center text-white shadow-lg shadow-indigo-500/25">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <span className="font-bold text-base tracking-tight text-white block leading-tight">Multimodal AI</span>
              <span className="text-[11px] text-slate-400">Local Consumer AI Agent</span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center bg-slate-900 border border-slate-800 p-1 rounded-xl text-xs font-semibold">
            <button
              onClick={() => setActiveTab('chat')}
              className={`px-4 py-1.5 rounded-lg flex items-center gap-2 transition ${
                activeTab === 'chat'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>Ask & Search</span>
            </button>
            <button
              onClick={() => {
                setActiveTab('upload');
                fetchHealthAndFiles();
              }}
              className={`px-4 py-1.5 rounded-lg flex items-center gap-2 transition ${
                activeTab === 'upload'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <CloudArrowUp className="w-3.5 h-3.5" />
              <span>Upload Files</span>
              <span className="px-1.5 py-0.2 rounded-full bg-slate-800 text-[10px] text-indigo-300">
                {fileList.length}
              </span>
            </button>
          </div>

          <span className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            {gpuStatus}
          </span>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-4xl mx-auto px-4 py-6 flex-1 flex flex-col w-full">
        
        {/* ============================================================= */}
        {/* TAB 1: CHAT & SEARCH                                         */}
        {/* ============================================================= */}
        {activeTab === 'chat' && (
          <div className="flex-1 flex flex-col w-full">
            {!displayedQuery && (
              <div className="text-center my-auto py-10">
                <div className="w-16 h-16 rounded-3xl bg-indigo-600/10 text-indigo-400 border border-indigo-500/20 flex items-center justify-center mx-auto text-2xl shadow-xl mb-4">
                  <Sparkles className="w-8 h-8" />
                </div>
                <h2 className="text-3xl font-bold text-white tracking-tight">Ask anything across your knowledge</h2>
                <p className="text-sm text-slate-400 mt-2 max-w-md mx-auto">
                  Ask questions, search for documents, listen to songs, or pinpoint video scenes.
                </p>

                <div className="flex flex-wrap justify-center gap-2 mt-6 max-w-xl mx-auto">
                  {[
                    { label: '🎵 Play Jhalak Dikhla Ja', text: 'Jhalak Dikhla Ja song and music audio' },
                    { label: '🚗 Car engine maintenance', text: 'How do I maintain my car engine and prevent wear?' },
                    { label: '🏎️ Red sports car in video', text: 'red sports car speeding on the highway' },
                    { label: '💻 Circuit board keyframe', text: 'silicon microchip and printed circuit board' },
                  ].map((item, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setQuery(item.text);
                      }}
                      className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 transition"
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {displayedQuery && (
              <div className="space-y-6 mb-8">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-xs text-slate-300 shrink-0 mt-0.5">
                    <User className="w-4 h-4" />
                  </div>
                  <div className="text-base font-semibold text-white pt-1">{displayedQuery}</div>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
                  <div className="flex items-center gap-2 text-xs font-bold text-indigo-400 uppercase tracking-wider mb-2.5">
                    <Sparkles className="w-3.5 h-3.5" /> AI Answer
                  </div>
                  {loading ? (
                    <p className="text-sm text-slate-400 italic">Searching files & generating grounded answer...</p>
                  ) : (
                    <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">{aiAnswer}</p>
                  )}
                </div>

                {sources.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      <span>Matching Evidence</span>
                      {elapsedMs && <span className="font-mono text-slate-500 text-[11px]">{elapsedMs}ms</span>}
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {sources.map((item, idx) => {
                        const mod = (item.modality || 'text').toLowerCase();
                        const filename = item.content ? item.content.split('/').pop() : item.source;

                        return (
                          <div key={idx} className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between hover:border-slate-700 transition shadow-lg">
                            <div>
                              <div className="flex items-center justify-between text-xs mb-2">
                                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                                  {mod.toUpperCase()}
                                </span>
                                <span className="text-xs font-mono text-indigo-400 font-semibold">
                                  {Math.round((item.similarity || 0) * 100)}% Match
                                </span>
                              </div>
                              <div className="text-sm font-bold text-white truncate">{item.title || filename}</div>
                              <div className="text-xs text-slate-500 truncate">{item.source}</div>
                            </div>

                            {mod === 'image' && (
                              <div className="mt-3 rounded-xl overflow-hidden border border-slate-800 aspect-video bg-black flex items-center justify-center">
                                <img src={`${apiUrl}/uploads/${filename}`} alt={item.title} className="w-full h-full object-cover" />
                              </div>
                            )}

                            {mod === 'audio' && (
                              <div className="mt-3 bg-slate-950 p-2 rounded-xl border border-slate-800">
                                <audio controls className="w-full h-9 rounded">
                                  <source src={`${apiUrl}/uploads/${filename}`} type="audio/mpeg" />
                                  <source src={`${apiUrl}/uploads/${filename}`} type="audio/wav" />
                                </audio>
                              </div>
                            )}

                            {mod === 'video' && (
                              <div className="mt-3 rounded-xl overflow-hidden border border-slate-800 aspect-video relative bg-black flex items-center justify-center">
                                <img src={`${apiUrl}/uploads/frames/${filename}`} alt={item.title} className="w-full h-full object-cover" />
                                <span className="absolute bottom-2 right-2 bg-black/80 font-mono text-[10px] text-white px-2 py-0.5 rounded-md border border-white/20">
                                  Keyframe
                                </span>
                              </div>
                            )}

                            {mod === 'text' && (
                              <div className="mt-3 p-3 rounded-xl bg-slate-950 text-xs text-slate-300 font-mono max-h-24 overflow-y-auto leading-relaxed border border-slate-800/80">
                                {item.content || item.title}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Bottom Search Bar */}
            <div className="mt-auto sticky bottom-4 z-40 pt-2">
              <div className="bg-slate-900 border border-slate-800 focus-within:border-indigo-500 rounded-2xl p-2 shadow-2xl transition">
                <form onSubmit={handleSearch} className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setActiveTab('upload')}
                    title="Upload new file"
                    className="w-10 h-10 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center justify-center transition shrink-0"
                  >
                    <FolderOpen className="w-4 h-4" />
                  </button>

                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask anything or search your files (e.g. 'Play Jhalak Dikhla Ja', 'car maintenance')..."
                    className="flex-1 bg-transparent text-sm placeholder-slate-500 outline-none text-slate-100 py-2.5 px-1"
                    required
                  />

                  <button
                    type="submit"
                    disabled={loading}
                    className="w-10 h-10 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white flex items-center justify-center transition shrink-0 shadow-md shadow-indigo-600/30"
                  >
                    <ArrowUp className="w-4 h-4" />
                  </button>
                </form>
              </div>

              <p className="text-[11px] text-slate-500 text-center mt-2">
                Runs 100% locally on your NVIDIA RTX 3050 Ti GPU • Completely Private & Free
              </p>
            </div>
          </div>
        )}

        {/* ============================================================= */}
        {/* TAB 2: UPLOAD & KNOWLEDGE LIBRARY VIEW                        */}
        {/* ============================================================= */}
        {activeTab === 'upload' && (
          <div className="space-y-6 w-full">
            <div className="bg-slate-900 border-2 border-dashed border-indigo-500/40 hover:border-indigo-500 rounded-3xl p-8 sm:p-12 text-center transition bg-indigo-950/10">
              <input
                type="file"
                id="tab-file-upload"
                className="hidden"
                accept=".txt,.md,.pdf,.png,.jpg,.jpeg,.wav,.mp3,.mp4,.mkv"
                onChange={(e) => {
                  if (e.target.files?.[0]) handleFileUpload(e.target.files[0]);
                }}
              />

              <div className="w-16 h-16 rounded-2xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 flex items-center justify-center mx-auto text-3xl mb-4">
                <CloudArrowUp className="w-8 h-8" />
              </div>

              <h3 className="text-xl font-bold text-white">Upload Any File to Knowledge Base</h3>
              <p className="text-sm text-slate-400 max-w-md mx-auto mt-2">
                Drag and drop files here, or click the button below to browse.
              </p>

              <button
                type="button"
                onClick={() => document.getElementById('tab-file-upload')?.click()}
                className="mt-6 px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold transition shadow-lg shadow-indigo-600/30 inline-flex items-center gap-2"
              >
                <FolderOpen className="w-4 h-4" />
                <span>Choose Files to Upload</span>
              </button>

              <div className="flex flex-wrap justify-center gap-2 mt-6 text-xs text-slate-400">
                <span className="px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700">📄 PDF / Text (.pdf, .txt, .md)</span>
                <span className="px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700">🖼️ Images (.png, .jpg)</span>
                <span className="px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700">🎵 Audio (.mp3, .wav)</span>
                <span className="px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700">🎬 Video (.mp4, .mkv)</span>
              </div>

              {uploadStatus && (
                <div className="mt-6 p-3 rounded-xl bg-indigo-950/90 border border-indigo-800 text-sm text-indigo-300 max-w-md mx-auto">
                  {uploadStatus}
                </div>
              )}
            </div>

            {/* List of Files */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <span>Your Indexed Files</span>
                  <span className="text-xs font-normal text-slate-400">({fileList.length} files)</span>
                </h3>
                <button
                  onClick={fetchHealthAndFiles}
                  className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium transition"
                >
                  <RefreshCw className="w-3.5 h-3.5" /> Refresh List
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                {fileList.map((f, i) => (
                  <div key={i} className="bg-slate-900 border border-slate-800 rounded-xl p-3 flex items-center justify-between hover:border-slate-700 transition">
                    <div className="min-w-0">
                      <p className="text-xs font-semibold text-white truncate" title={f.name}>{f.name}</p>
                      <p className="text-[10px] text-slate-500">{f.size} • {f.modality.toUpperCase()}</p>
                    </div>
                    <a
                      href={`${apiUrl}${f.url}`}
                      target="_blank"
                      rel="noreferrer"
                      className="w-7 h-7 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white flex items-center justify-center transition shrink-0 ml-2"
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

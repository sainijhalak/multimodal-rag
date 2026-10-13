'use client';

import React, { useState, useEffect } from 'react';
import { Sparkles, Plus, ArrowUp, User, Bot } from 'lucide-react';

interface SourceItem {
  title: string;
  content: string;
  source: string;
  modality: string;
  similarity: number;
}

export default function Home() {
  const [apiUrl, setApiUrl] = useState<string>(
    process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  );
  const [gpuStatus, setGpuStatus] = useState<string>('RTX 3050 Ti • Local');
  const [query, setQuery] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [displayedQuery, setDisplayedQuery] = useState<string | null>(null);
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${apiUrl}/health/`);
        if (res.ok) {
          const data = await res.json();
          const dev = data.device ? data.device.replace('NVIDIA GeForce ', '').replace(' Laptop GPU', '') : 'CUDA';
          setGpuStatus(`${dev} • ${data.total_documents_indexed || 0} items`);
        }
      } catch (e) {
        console.warn('Backend not reached yet');
      }
    };
    fetchHealth();
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
    setUploadStatus(`Indexing ${file.name}...`);
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
      <header className="border-b border-slate-900 bg-slate-950/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-3xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-md shadow-indigo-600/20">
              <Sparkles className="w-4 h-4" />
            </div>
            <span className="font-bold text-sm tracking-tight text-white">Multimodal AI</span>
          </div>

          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
            {gpuStatus}
          </span>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-3xl mx-auto px-4 py-8 flex-1 flex flex-col w-full">
        
        {/* Welcome Screen */}
        {!displayedQuery && (
          <div className="text-center my-auto py-12">
            <div className="w-16 h-16 rounded-3xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center mx-auto text-white text-2xl shadow-xl shadow-indigo-500/20 mb-5">
              <Sparkles className="w-8 h-8" />
            </div>
            <h2 className="text-3xl font-bold text-white tracking-tight">Ask anything across your media</h2>
            <p className="text-sm text-slate-400 mt-2 max-w-md mx-auto">
              Search documents, listen to audio, inspect images, and pinpoint video scenes—all in one place.
            </p>

            <div className="flex flex-wrap justify-center gap-2 mt-6 max-w-lg mx-auto">
              {[
                { label: '🚗 Car maintenance advice', text: 'How do I maintain my car engine and prevent wear?' },
                { label: '🏎️ Red sports car', text: 'red sports car speeding on the highway' },
                { label: '🚨 Siren alarm sound', text: 'emergency warning siren beep and alarm sound' },
                { label: '💻 Circuit board in video', text: 'silicon microchip and printed circuit board' },
              ].map((item, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(item.text)}
                  className="px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 transition"
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Unified Results Area */}
        {displayedQuery && (
          <div className="space-y-6 mb-8">
            <div className="flex items-start gap-3">
              <div className="w-7 h-7 rounded-full bg-slate-800 flex items-center justify-center text-xs text-slate-400 shrink-0 mt-0.5">
                <User className="w-4 h-4" />
              </div>
              <div className="text-base font-semibold text-white pt-0.5">{displayedQuery}</div>
            </div>

            {/* AI Answer */}
            <div className="bg-slate-900/80 rounded-2xl p-5 border border-slate-800 shadow-lg">
              <div className="flex items-center gap-2 text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-2">
                <Bot className="w-4 h-4" /> AI Answer
              </div>
              {loading ? (
                <p className="text-sm text-slate-400 italic">Searching files and generating answer...</p>
              ) : (
                <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">{aiAnswer}</p>
              )}
            </div>

            {/* Evidence Cards */}
            {sources.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  <span>Matching Evidence</span>
                  {elapsedMs && <span className="font-mono text-slate-500 text-[11px]">{elapsedMs}ms</span>}
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {sources.map((item, idx) => {
                    const mod = (item.modality || 'text').toLowerCase();
                    const filename = item.content ? item.content.split('/').pop() : item.source;

                    return (
                      <div key={idx} className="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col justify-between hover:border-slate-700 transition">
                        <div>
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                              {mod.toUpperCase()}
                            </span>
                            <span className="text-[11px] font-mono text-slate-400">
                              Match: {Math.round((item.similarity || 0) * 100)}%
                            </span>
                          </div>
                          <div className="text-xs font-semibold text-white truncate">{item.title || filename}</div>
                          <div className="text-[11px] text-slate-500 truncate">{item.source}</div>
                        </div>

                        {mod === 'image' && (
                          <div className="mt-2.5 rounded-lg overflow-hidden border border-slate-800 aspect-video bg-black flex items-center justify-center">
                            <img src={`${apiUrl}/uploads/${filename}`} alt={item.title} className="w-full h-full object-cover" />
                          </div>
                        )}

                        {mod === 'audio' && (
                          <div className="mt-2.5">
                            <audio controls className="w-full h-8 rounded bg-slate-950">
                              <source src={`${apiUrl}/uploads/${filename}`} type="audio/wav" />
                            </audio>
                          </div>
                        )}

                        {mod === 'video' && (
                          <div className="mt-2.5 rounded-lg overflow-hidden border border-slate-800 aspect-video relative bg-black flex items-center justify-center">
                            <img src={`${apiUrl}/uploads/frames/${filename}`} alt={item.title} className="w-full h-full object-cover" />
                            <span className="absolute bottom-1.5 right-1.5 bg-black/80 font-mono text-[10px] text-white px-1.5 py-0.5 rounded border border-white/20">
                              Keyframe
                            </span>
                          </div>
                        )}

                        {mod === 'text' && (
                          <div className="mt-2.5 p-2 rounded bg-slate-950 text-[11px] text-slate-300 font-mono max-h-20 overflow-y-auto leading-relaxed border border-slate-800/80">
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
        <div className="mt-auto sticky bottom-4 z-40">
          {uploadStatus && (
            <div className="mb-2 px-3 py-2 rounded-xl bg-indigo-950/80 border border-indigo-800 text-xs text-indigo-300 text-center shadow-lg">
              {uploadStatus}
            </div>
          )}

          <div className="bg-slate-900 border border-slate-800 focus-within:border-indigo-500 rounded-2xl p-2 shadow-2xl transition">
            <form onSubmit={handleSearch} className="flex items-center gap-2">
              <input
                type="file"
                id="next-file-upload"
                className="hidden"
                accept=".txt,.md,.pdf,.png,.jpg,.jpeg,.wav,.mp3,.mp4,.mkv"
                onChange={(e) => {
                  if (e.target.files?.[0]) handleFileUpload(e.target.files[0]);
                }}
              />
              <button
                type="button"
                onClick={() => document.getElementById('next-file-upload')?.click()}
                title="Attach any file (Documents, Photos, Audio, Video)"
                className="w-10 h-10 rounded-xl hover:bg-slate-800 text-slate-400 hover:text-slate-200 flex items-center justify-center transition shrink-0"
              >
                <Plus className="w-5 h-5" />
              </button>

              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question or search your files..."
                className="flex-1 bg-transparent text-sm placeholder-slate-500 outline-none text-slate-100 py-2"
                required
              />

              <button
                type="submit"
                disabled={loading}
                className="w-10 h-10 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white flex items-center justify-center transition shrink-0 shadow-md shadow-indigo-600/30"
              >
                <ArrowUp className="w-5 h-5" />
              </button>
            </form>
          </div>

          <p className="text-[11px] text-slate-500 text-center mt-2">
            Powered by Google EmbeddingGemma-2 & Qwen2.5 on your local RTX 3050 Ti GPU • 100% Free
          </p>
        </div>

      </main>
    </div>
  );
}

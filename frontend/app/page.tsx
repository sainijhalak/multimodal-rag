'use client';

import React, { useState, useEffect } from 'react';
import { 
  Brain, Database, CloudArrowUp, FileText, Image as ImageIcon, 
  Volume2, Film, Sparkles, Search, ArrowRight, Bot, Shapes, ChevronRight 
} from 'lucide-react';

interface Telemetry {
  device: string;
  vram_allocated_mb: number;
  total_documents_indexed: number;
}

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
  const [telemetry, setTelemetry] = useState<Telemetry>({
    device: 'Detecting GPU...',
    vram_allocated_mb: 0,
    total_documents_indexed: 0,
  });
  const [mode, setMode] = useState<'ask' | 'search'>('ask');
  const [selectedModality, setSelectedModality] = useState<string>('');
  const [query, setQuery] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  // Fetch telemetry on load & interval
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${apiUrl}/health/`);
        if (res.ok) {
          const data = await res.json();
          setTelemetry({
            device: data.device || 'CUDA',
            vram_allocated_mb: data.vram_allocated_mb || 0,
            total_documents_indexed: data.total_documents_indexed || 0,
          });
        }
      } catch (e) {
        console.warn('Backend not reached yet at', apiUrl);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, [apiUrl]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setAiAnswer(null);
    setSources([]);
    const start = performance.now();

    try {
      if (mode === 'ask') {
        const res = await fetch(`${apiUrl}/query/ask`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query.trim(),
            top_k: 4,
            modality: selectedModality || null,
          }),
        });
        const data = await res.json();
        setAiAnswer(data.answer || 'No direct answer returned.');
        setSources(data.sources || []);
      } else {
        const res = await fetch(`${apiUrl}/query/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query.trim(),
            top_k: 4,
            modality: selectedModality || null,
          }),
        });
        const data = await res.json();
        setSources(data.results || []);
      }
    } catch (err: any) {
      alert('Request failed: ' + err.message);
    } finally {
      setElapsedMs(Math.round(performance.now() - start));
      setLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setUploadStatus(`Ingesting ${file.name}...`);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${apiUrl}/ingest/file`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`✓ Successfully indexed ${data.filename} (${data.chunks_created} vectors)!`);
      } else {
        setUploadStatus(`Error: ${data.detail || 'Upload failed'}`);
      }
    } catch (err: any) {
      setUploadStatus(`Upload failed: ${err.message}`);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="glass sticky top-0 z-50 border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg leading-tight tracking-tight flex items-center gap-2">
                Multimodal RAG Agent
                <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                  100% Free & Offline
                </span>
              </h1>
              <p className="text-xs text-slate-400">
                Google EmbeddingGemma-2 • Qwen2.5-0.5B • LanceDB
              </p>
            </div>
          </div>

          {/* Telemetry Stats */}
          <div className="flex items-center space-x-3 text-xs">
            <div className="glass px-3 py-1.5 rounded-lg flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-slate-300 font-medium">{telemetry.device}</span>
              <span className="text-slate-500">|</span>
              <span className="text-indigo-400 font-mono">
                {telemetry.vram_allocated_mb} MB VRAM
              </span>
            </div>
            <div className="glass px-3 py-1.5 rounded-lg flex items-center gap-2">
              <Database className="w-3.5 h-3.5 text-purple-400" />
              <span className="text-slate-300 font-medium">
                {telemetry.total_documents_indexed} items
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 w-full">
        {/* Left Sidebar (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          {/* File Ingestion */}
          <div className="glass rounded-2xl p-5 shadow-xl">
            <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider mb-3 flex items-center gap-2">
              <CloudArrowUp className="w-4 h-4 text-indigo-400" /> File Ingestion
            </h2>
            <div
              onClick={() => document.getElementById('next-file-input')?.click()}
              className="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-6 text-center transition-all cursor-pointer bg-slate-900/50 hover:bg-indigo-950/20"
            >
              <input
                type="file"
                id="next-file-input"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.[0]) handleFileUpload(e.target.files[0]);
                }}
              />
              <div className="w-12 h-12 mx-auto rounded-full bg-indigo-500/10 text-indigo-400 flex items-center justify-center mb-3">
                <CloudArrowUp className="w-6 h-6" />
              </div>
              <p className="text-sm font-medium text-slate-300">Drop files or click to upload</p>
              <p className="text-xs text-slate-500 mt-1">Text, Image, Audio & Video</p>
              <div className="flex justify-center gap-1.5 mt-3 text-[11px] text-slate-400">
                <span className="px-1.5 py-0.5 rounded bg-slate-800">.txt</span>
                <span className="px-1.5 py-0.5 rounded bg-slate-800">.png</span>
                <span className="px-1.5 py-0.5 rounded bg-slate-800">.wav</span>
                <span className="px-1.5 py-0.5 rounded bg-slate-800">.mp4</span>
              </div>
            </div>
            {uploadStatus && (
              <p className="mt-3 text-xs text-indigo-300">{uploadStatus}</p>
            )}
          </div>

          {/* Modality Filter */}
          <div className="glass rounded-2xl p-5 shadow-xl">
            <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider mb-3 flex items-center gap-2">
              <Shapes className="w-4 h-4 text-purple-400" /> Modality Filter
            </h2>
            <div className="grid grid-cols-2 gap-2 text-xs font-medium">
              <button
                onClick={() => setSelectedModality('')}
                className={`px-3 py-2 rounded-xl border flex items-center gap-2 transition ${
                  selectedModality === ''
                    ? 'border-indigo-500/50 bg-indigo-500/20 text-indigo-300'
                    : 'border-slate-800 bg-slate-900/60 text-slate-400 hover:text-slate-200'
                }`}
              >
                <Shapes className="w-3.5 h-3.5" /> All Modalities
              </button>
              <button
                onClick={() => setSelectedModality('text')}
                className={`px-3 py-2 rounded-xl border flex items-center gap-2 transition ${
                  selectedModality === 'text'
                    ? 'border-indigo-500/50 bg-indigo-500/20 text-indigo-300'
                    : 'border-slate-800 bg-slate-900/60 text-slate-400 hover:text-slate-200'
                }`}
              >
                <FileText className="w-3.5 h-3.5 text-blue-400" /> Documents
              </button>
              <button
                onClick={() => setSelectedModality('image')}
                className={`px-3 py-2 rounded-xl border flex items-center gap-2 transition ${
                  selectedModality === 'image'
                    ? 'border-indigo-500/50 bg-indigo-500/20 text-indigo-300'
                    : 'border-slate-800 bg-slate-900/60 text-slate-400 hover:text-slate-200'
                }`}
              >
                <ImageIcon className="w-3.5 h-3.5 text-emerald-400" /> Images
              </button>
              <button
                onClick={() => setSelectedModality('audio')}
                className={`px-3 py-2 rounded-xl border flex items-center gap-2 transition ${
                  selectedModality === 'audio'
                    ? 'border-indigo-500/50 bg-indigo-500/20 text-indigo-300'
                    : 'border-slate-800 bg-slate-900/60 text-slate-400 hover:text-slate-200'
                }`}
              >
                <Volume2 className="w-3.5 h-3.5 text-amber-400" /> Audio
              </button>
              <button
                onClick={() => setSelectedModality('video')}
                className={`col-span-2 px-3 py-2 rounded-xl border flex items-center gap-2 transition ${
                  selectedModality === 'video'
                    ? 'border-indigo-500/50 bg-indigo-500/20 text-indigo-300'
                    : 'border-slate-800 bg-slate-900/60 text-slate-400 hover:text-slate-200'
                }`}
              >
                <Film className="w-3.5 h-3.5 text-pink-400" /> Video Keyframes (1 FPS)
              </button>
            </div>
          </div>

          {/* Quick Prompts */}
          <div className="glass rounded-2xl p-5 shadow-xl">
            <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider mb-3 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-400" /> Quick Demonstrations
            </h2>
            <div className="space-y-2 text-xs">
              {[
                { title: '🚗 Vehicle maintenance advice', text: 'How do I prevent my vehicle engine from wearing out?' },
                { title: '🏎️ Find sports car in video & image', text: 'red sports car speeding on the highway' },
                { title: '🚨 Retrieve audio siren clip', text: 'emergency warning siren beep and alarm sound' },
                { title: '💻 Circuit board in video at 00:02', text: 'silicon microchip and printed circuit board' },
              ].map((item, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(item.text)}
                  className="w-full text-left p-2.5 rounded-lg bg-slate-900/50 hover:bg-slate-800/80 border border-slate-800 text-slate-300 transition flex items-center justify-between"
                >
                  <span>{item.title}</span>
                  <ChevronRight className="w-3 h-3 text-slate-600" />
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Main Column (8 cols) */}
        <div className="lg:col-span-8 flex flex-col space-y-6">
          {/* Query Box */}
          <div className="glass rounded-2xl p-5 shadow-2xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setMode('ask')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1 transition ${
                    mode === 'ask'
                      ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                      : 'bg-slate-800/60 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Sparkles className="w-3 h-3 mr-1" /> Ask AI (Full RAG)
                </button>
                <button
                  onClick={() => setMode('search')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1 transition ${
                    mode === 'search'
                      ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                      : 'bg-slate-800/60 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Search className="w-3 h-3 mr-1" /> Similarity Search
                </button>
              </div>
              <span className="text-xs text-slate-500 font-mono">Top-K: 4</span>
            </div>

            <form onSubmit={handleSearch} className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question or search across text, images, audio, and video..."
                className="w-full bg-slate-900/80 border border-slate-700/80 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 rounded-xl px-4 py-3.5 pr-28 text-sm placeholder-slate-500 outline-none transition text-slate-100"
                required
              />
              <button
                type="submit"
                disabled={loading}
                className="absolute right-2 top-2 bottom-2 px-5 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 rounded-lg text-xs font-semibold text-white transition flex items-center gap-2 shadow-md disabled:opacity-50"
              >
                <span>{loading ? 'Processing...' : mode === 'ask' ? 'Ask RAG' : 'Search'}</span>
                <ArrowRight className="w-3 h-3" />
              </button>
            </form>
          </div>

          {/* Results Area */}
          <div className="space-y-6 flex-1">
            {!aiAnswer && sources.length === 0 && !loading && (
              <div className="glass rounded-2xl p-12 text-center border border-slate-800">
                <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 text-indigo-400 mx-auto flex items-center justify-center text-2xl mb-4">
                  <Brain className="w-8 h-8" />
                </div>
                <h3 className="text-lg font-semibold text-slate-200">Multimodal Agent Ready</h3>
                <p className="text-sm text-slate-400 max-w-md mx-auto mt-2">
                  Ask any question to retrieve and synthesize answers across text manuals, visual images, synthesized audio sirens, and 1 FPS video frames!
                </p>
              </div>
            )}

            {/* AI Generative Answer Card */}
            {aiAnswer && (
              <div className="glass rounded-2xl p-6 border border-indigo-500/20 shadow-2xl">
                <div className="flex items-center gap-2 text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-3">
                  <Bot className="w-4 h-4" /> Grounded AI Response (Qwen2.5)
                </div>
                <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap font-normal">
                  {aiAnswer}
                </div>
              </div>
            )}

            {/* Retrieved Multimodal Sources */}
            {sources.length > 0 && (
              <div className="space-y-4">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  <span>Retrieved Multimodal Evidence</span>
                  {elapsedMs && <span className="font-mono text-slate-500">Completed in {elapsedMs}ms</span>}
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {sources.map((item, idx) => {
                    const mod = (item.modality || 'text').toLowerCase();
                    const filename = item.content ? item.content.split('/').pop() : item.source;

                    let badgeColor = 'bg-blue-500/10 text-blue-400 border-blue-500/20';
                    if (mod === 'image') badgeColor = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
                    if (mod === 'audio') badgeColor = 'bg-amber-500/10 text-amber-400 border-amber-500/20';
                    if (mod === 'video') badgeColor = 'bg-pink-500/10 text-pink-400 border-pink-500/20';

                    return (
                      <div key={idx} className="glass-card rounded-xl p-4 flex flex-col justify-between hover:border-slate-700 transition">
                        <div>
                          <div className="flex items-center justify-between text-xs mb-2">
                            <span className={`px-2 py-0.5 rounded-full border text-[10px] font-semibold flex items-center gap-1.5 ${badgeColor}`}>
                              {mod.toUpperCase()}
                            </span>
                            <span className="text-slate-400 font-mono text-[11px]">
                              Sim: {(item.similarity || 0).toFixed(4)}
                            </span>
                          </div>
                          <h4 className="text-sm font-semibold text-slate-100 truncate">{item.title || filename}</h4>
                          <p className="text-xs text-slate-500 truncate">Source: {item.source}</p>
                        </div>

                        {mod === 'image' && (
                          <div className="mt-3 rounded-lg overflow-hidden border border-slate-800 bg-slate-900 aspect-video flex items-center justify-center">
                            <img src={`${apiUrl}/uploads/${filename}`} alt={item.title} className="w-full h-full object-cover" />
                          </div>
                        )}

                        {mod === 'audio' && (
                          <div className="mt-3">
                            <audio controls className="w-full h-8 rounded bg-slate-900 border border-slate-800">
                              <source src={`${apiUrl}/uploads/${filename}`} type="audio/wav" />
                            </audio>
                          </div>
                        )}

                        {mod === 'video' && (
                          <div className="mt-3 rounded-lg overflow-hidden border border-slate-800 bg-slate-900 aspect-video relative flex items-center justify-center">
                            <img src={`${apiUrl}/uploads/frames/${filename}`} alt={item.title} className="w-full h-full object-cover" />
                            <span className="absolute bottom-2 right-2 bg-black/80 text-white font-mono text-[10px] px-2 py-0.5 rounded border border-white/20">
                              Keyframe
                            </span>
                          </div>
                        )}

                        {mod === 'text' && (
                          <div className="mt-3 p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 text-xs text-slate-300 font-mono max-h-24 overflow-y-auto leading-relaxed">
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
        </div>
      </main>
    </div>
  );
}

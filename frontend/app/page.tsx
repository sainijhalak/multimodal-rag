'use client';

import React, { useState, useRef, useEffect } from 'react';
import { 
  Sparkles, Paperclip, ArrowUp, X, Check, Loader2, 
  RotateCcw, Music, Video, Image as ImageIcon, FileText, Upload
} from 'lucide-react';

interface SourceItem {
  title: string;
  content: string;
  source: string;
  modality: string;
  similarity: number;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content?: string;
  sources?: SourceItem[];
  fileUpload?: {
    name: string;
    status: 'uploading' | 'success' | 'error';
  };
}

export default function Home() {
  const [apiUrl, setApiUrl] = useState<string>(
    process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  );
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>('');
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const dragCounter = useRef<number>(0);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // Drag and drop listeners
  useEffect(() => {
    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      dragCounter.current += 1;
      setIsDragging(true);
    };

    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault();
      dragCounter.current -= 1;
      if (dragCounter.current <= 0) {
        dragCounter.current = 0;
        setIsDragging(false);
      }
    };

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
    };

    const handleDrop = (e: DragEvent) => {
      e.preventDefault();
      dragCounter.current = 0;
      setIsDragging(false);
      if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
        setAttachedFile(e.dataTransfer.files[0]);
      }
    };

    window.addEventListener('dragenter', handleDragEnter);
    window.addEventListener('dragleave', handleDragLeave);
    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('drop', handleDrop);

    return () => {
      window.removeEventListener('dragenter', handleDragEnter);
      window.removeEventListener('dragleave', handleDragLeave);
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('drop', handleDrop);
    };
  }, []);

  const handleSendMessage = async (customPrompt?: string) => {
    const text = (customPrompt !== undefined ? customPrompt : input).trim();
    if (!text && !attachedFile) return;

    const fileToUpload = attachedFile;
    setAttachedFile(null);
    setInput('');

    // If there is an attached file, upload & index it first
    if (fileToUpload) {
      const uploadMsgId = 'upload-' + Date.now();
      setMessages((prev) => [
        ...prev,
        {
          id: uploadMsgId,
          role: 'system',
          fileUpload: { name: fileToUpload.name, status: 'uploading' },
        },
      ]);

      const formData = new FormData();
      formData.append('file', fileToUpload);

      try {
        const res = await fetch(`${apiUrl}/ingest/file`, {
          method: 'POST',
          body: formData,
        });
        if (res.ok) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === uploadMsgId
                ? {
                    ...m,
                    fileUpload: { name: fileToUpload.name, status: 'success' },
                  }
                : m
            )
          );
        } else {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === uploadMsgId
                ? {
                    ...m,
                    fileUpload: { name: fileToUpload.name, status: 'error' },
                  }
                : m
            )
          );
        }
      } catch {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === uploadMsgId
              ? {
                  ...m,
                  fileUpload: { name: fileToUpload.name, status: 'error' },
                }
              : m
          )
        );
      }
    }

    if (!text) return;

    // Add user message
    const userMsgId = 'user-' + Date.now();
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: 'user', content: text },
    ]);

    setLoading(true);

    try {
      const res = await fetch(`${apiUrl}/query/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, top_k: 4 }),
      });
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          id: 'bot-' + Date.now(),
          role: 'assistant',
          content: data.answer || 'I could not find an answer in your files.',
          sources: (data.sources || []).filter(
            (s: SourceItem) => s.similarity >= 0.55
          ),
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: 'bot-' + Date.now(),
          role: 'assistant',
          content: `Error connecting to AI backend: ${err.message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const resetChat = () => {
    setMessages([]);
    setAttachedFile(null);
    setInput('');
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#0f1117] text-slate-100 selection:bg-indigo-500 selection:text-white relative">
      {/* Fullscreen Drag & Drop Overlay */}
      {isDragging && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center border-2 border-dashed border-indigo-500 m-4 rounded-3xl pointer-events-none">
          <div className="w-16 h-16 rounded-2xl bg-indigo-600/20 text-indigo-400 flex items-center justify-center text-3xl mb-4 border border-indigo-500/30 animate-bounce">
            <Upload className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-bold text-white">Drop file to upload & index</h3>
          <p className="text-sm text-slate-400 mt-1">Supports PDF, Audio (.mp3, .wav), Video (.mp4), and Images</p>
        </div>
      )}

      {/* Header */}
      <header className="border-b border-slate-800/60 bg-[#0f1117]/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-3xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center text-white text-xs shadow-md shadow-indigo-600/30">
              <Sparkles className="w-4 h-4" />
            </div>
            <span className="font-semibold text-sm tracking-tight text-white">Multimodal AI</span>
          </div>

          <button
            onClick={resetChat}
            className="text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded-lg hover:bg-slate-800 transition flex items-center gap-1.5 border border-transparent hover:border-slate-700"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>New Chat</span>
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-3xl mx-auto px-4 py-6 flex-1 flex flex-col w-full">
        <div className="flex-1 space-y-6 pb-28">
          {messages.length === 0 && (
            <div className="text-center my-auto py-12">
              <div className="w-14 h-14 rounded-2xl bg-indigo-600/10 text-indigo-400 flex items-center justify-center mx-auto text-2xl mb-4 border border-indigo-500/20 shadow-lg shadow-indigo-500/10">
                <Sparkles className="w-6 h-6" />
              </div>
              <h2 className="text-2xl font-bold text-white tracking-tight">How can I help you today?</h2>
              <p className="text-xs text-slate-400 mt-2 max-w-md mx-auto leading-relaxed">
                Ask questions across your documents, listen to songs, search photos, or pinpoint video scenes.
              </p>

              {/* Upload Drop Zone Card in Welcome Screen */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className="mt-6 mx-auto max-w-md p-4 rounded-2xl border border-dashed border-slate-800 hover:border-indigo-500/50 bg-slate-900/50 hover:bg-slate-900 cursor-pointer transition flex items-center gap-3.5 group text-left"
              >
                <div className="w-10 h-10 rounded-xl bg-slate-800 group-hover:bg-indigo-600/20 text-slate-400 group-hover:text-indigo-400 flex items-center justify-center text-base shrink-0 transition">
                  <Upload className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-semibold text-slate-200 group-hover:text-white transition">
                    Upload any document, audio, image, or video
                  </div>
                  <div className="text-[11px] text-slate-500 truncate mt-0.5">
                    Click to browse, or drag & drop files anywhere
                  </div>
                </div>
                <span className="text-xs text-slate-500 group-hover:text-indigo-400 transition font-medium">Browse</span>
              </div>

              {/* Suggestions */}
              <div className="flex flex-wrap justify-center gap-2 mt-6 max-w-md mx-auto">
                {[
                  'Play Jhalak Dikhla Ja song',
                  'How do I maintain my car engine and prevent wear?',
                  'Show me the red sports car',
                ].map((prompt, i) => (
                  <button
                    key={i}
                    onClick={() => handleSendMessage(prompt)}
                    className="px-3.5 py-1.5 rounded-full bg-slate-900/80 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 transition"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Conversation history */}
          {messages.map((msg) => {
            if (msg.role === 'system' && msg.fileUpload) {
              return (
                <div key={msg.id} className="flex justify-end">
                  {msg.fileUpload.status === 'uploading' && (
                    <div className="bg-indigo-600/20 border border-indigo-500/30 text-indigo-200 text-xs px-3.5 py-2 rounded-xl flex items-center gap-2">
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Uploading & indexing <b>{msg.fileUpload.name}</b>...</span>
                    </div>
                  )}
                  {msg.fileUpload.status === 'success' && (
                    <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs px-3.5 py-2 rounded-xl flex items-center gap-2">
                      <Check className="w-3.5 h-3.5" />
                      <span>Indexed <b>{msg.fileUpload.name}</b> into your knowledge base</span>
                    </div>
                  )}
                  {msg.fileUpload.status === 'error' && (
                    <div className="bg-red-500/10 border border-red-500/20 text-red-300 text-xs px-3.5 py-2 rounded-xl">
                      Failed to upload {msg.fileUpload.name}
                    </div>
                  )}
                </div>
              );
            }

            if (msg.role === 'user') {
              return (
                <div key={msg.id} className="flex justify-end">
                  <div className="bg-indigo-600 text-white text-sm px-4 py-2.5 rounded-2xl rounded-tr-sm max-w-[85%] leading-relaxed shadow-sm">
                    {msg.content}
                  </div>
                </div>
              );
            }

            return (
              <div key={msg.id} className="flex items-start gap-3">
                <div className="w-7 h-7 rounded-lg bg-slate-800 text-indigo-400 flex items-center justify-center shrink-0 text-xs mt-1">
                  <Sparkles className="w-3.5 h-3.5" />
                </div>
                <div className="flex-1 space-y-3">
                  <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                    {msg.content}
                  </div>

                  {msg.sources && msg.sources.length > 0 && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-2">
                      {msg.sources.map((item, idx) => {
                        const mod = (item.modality || 'text').toLowerCase();
                        const filename = item.content ? item.content.split('/').pop() : item.source;

                        return (
                          <div key={idx} className="bg-slate-900 border border-slate-800/80 rounded-xl p-3 flex flex-col justify-between">
                            {mod === 'audio' && (
                              <>
                                <div className="flex items-center justify-between text-xs mb-2">
                                  <span className="font-medium text-slate-200 truncate">{item.title || filename}</span>
                                  <span className="text-[10px] text-amber-400 font-mono">AUDIO</span>
                                </div>
                                <audio controls className="w-full h-8 rounded mt-1 bg-slate-950">
                                  <source src={`${apiUrl}/uploads/${filename}`} type="audio/mpeg" />
                                  <source src={`${apiUrl}/uploads/${filename}`} type="audio/wav" />
                                </audio>
                              </>
                            )}

                            {mod === 'image' && (
                              <>
                                <div className="flex items-center justify-between text-xs mb-1.5">
                                  <span className="font-medium text-slate-200 truncate">{item.title || filename}</span>
                                  <span className="text-[10px] text-emerald-400 font-mono">IMAGE</span>
                                </div>
                                <div className="rounded-lg overflow-hidden border border-slate-800 aspect-video bg-black flex items-center justify-center">
                                  <img src={`${apiUrl}/uploads/${filename}`} alt={item.title} className="w-full h-full object-cover" />
                                </div>
                              </>
                            )}

                            {mod === 'video' && (
                              <>
                                <div className="flex items-center justify-between text-xs mb-1.5">
                                  <span className="font-medium text-slate-200 truncate">{item.title || filename}</span>
                                  <span className="text-[10px] text-pink-400 font-mono">VIDEO</span>
                                </div>
                                <div className="rounded-lg overflow-hidden border border-slate-800 aspect-video bg-black relative flex items-center justify-center">
                                  <img src={`${apiUrl}/uploads/frames/${filename}`} alt={item.title} className="w-full h-full object-cover" />
                                  <span className="absolute bottom-1.5 right-1.5 bg-black/80 font-mono text-[9px] text-white px-1.5 py-0.5 rounded">
                                    Keyframe
                                  </span>
                                </div>
                              </>
                            )}

                            {mod === 'text' && (
                              <>
                                <div className="flex items-center justify-between text-xs mb-1">
                                  <span className="font-medium text-slate-200 truncate">{item.title || filename}</span>
                                  <span className="text-[10px] text-blue-400 font-mono">DOC</span>
                                </div>
                                <p className="text-[11px] text-slate-400 line-clamp-3 leading-relaxed mt-1 font-mono">
                                  {item.content || item.title}
                                </p>
                              </>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          {loading && (
            <div className="flex items-start gap-3">
              <div className="w-7 h-7 rounded-lg bg-slate-800 text-indigo-400 flex items-center justify-center shrink-0 text-xs mt-1">
                <Sparkles className="w-3.5 h-3.5" />
              </div>
              <div className="text-sm text-slate-400 italic flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> Thinking...
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Fixed Bottom Input Bar */}
        <div className="fixed bottom-0 left-0 right-0 bg-gradient-to-t from-[#0f1117] via-[#0f1117] to-transparent pt-6 pb-4 z-30">
          <div className="max-w-3xl mx-auto px-4">
            {/* Attached file preview chip */}
            {attachedFile && (
              <div className="mb-2 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-indigo-500/40 text-xs text-indigo-300 shadow-sm animate-fade-in">
                <Paperclip className="w-3 h-3" />
                <span className="font-medium truncate max-w-xs">{attachedFile.name}</span>
                <button onClick={() => setAttachedFile(null)} className="text-slate-400 hover:text-white ml-1">
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}

            <div className="bg-slate-900/90 backdrop-blur border border-slate-800 focus-within:border-indigo-500 rounded-2xl p-2 shadow-2xl transition flex items-center gap-2">
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept=".txt,.md,.pdf,.png,.jpg,.jpeg,.wav,.mp3,.mp4,.mkv"
                onChange={(e) => {
                  if (e.target.files?.[0]) {
                    setAttachedFile(e.target.files[0]);
                  }
                }}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                title="Attach a file (PDF, MP3, MP4, JPG)"
                className="w-9 h-9 rounded-xl hover:bg-slate-800 text-slate-400 hover:text-white flex items-center justify-center transition shrink-0"
              >
                <Paperclip className="w-4 h-4" />
              </button>

              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder="Message Multimodal AI, or attach a file..."
                className="flex-1 bg-transparent text-sm placeholder-slate-500 outline-none text-slate-100 py-2 px-1"
              />

              <button
                type="button"
                onClick={() => handleSendMessage()}
                disabled={loading}
                className="w-9 h-9 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white flex items-center justify-center transition shrink-0 shadow-md shadow-indigo-600/30"
              >
                <ArrowUp className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

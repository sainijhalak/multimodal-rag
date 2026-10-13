import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Multimodal RAG Agent | Offline Consumer AI',
  description: '100% Local Multimodal RAG with Google EmbeddingGemma-2 & LanceDB on consumer GPU',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased selection:bg-indigo-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}

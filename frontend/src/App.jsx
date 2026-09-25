import { useState } from 'react';

const API_BASE_URL = 'http://localhost:8015';
const CHAT_URL = `${API_BASE_URL}/chat`;
const UPLOAD_URL = `${API_BASE_URL}/documents/upload`;

export default function App() {
  const [question, setQuestion] = useState('');
  const [context, setContext] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState('');

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadMessage('Please select a PDF file first.');
      return;
    }

    setUploading(true);
    setUploadMessage('');
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(UPLOAD_URL, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      setUploadMessage(`Indexed ${data.filename} successfully. ${data.chunks} chunks added.`);
    } catch (err) {
      setUploadMessage(err.message || 'Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!question.trim()) {
      setError('Please enter a question.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch(CHAT_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          context: context || null,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Something went wrong');
      }

      setAnswer(data.answer || 'No answer returned.');
    } catch (err) {
      setError(
        err.message || 'Could not reach the backend. Please make sure FastAPI is running on localhost:8015.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="chat-card">
        <h1>GovAssist AI</h1>

        <div className="upload-box">
          <label htmlFor="pdf-upload">Upload PDF</label>
          <input
            id="pdf-upload"
            type="file"
            accept="application/pdf"
            onChange={(event) => setSelectedFile(event.target.files[0])}
          />
          <button type="button" onClick={handleUpload} disabled={uploading || !selectedFile}>
            {uploading ? 'Uploading...' : 'Index PDF'}
          </button>
          {uploadMessage && <div className="message info">{uploadMessage}</div>}
        </div>

        <form onSubmit={handleSubmit} className="chat-form">
          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            rows="4"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about a government document or policy..."
          />

          <label htmlFor="context">Optional context</label>
          <textarea
            id="context"
            rows="5"
            value={context}
            onChange={(event) => setContext(event.target.value)}
            placeholder="Paste relevant document text here if you want to provide context manually."
          />

          <button type="submit" disabled={loading}>
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </form>

        {error && <div className="message error">{error}</div>}

        <div className="answer-box">
          <h2>Answer</h2>
          <p>{answer || 'Your answer will appear here.'}</p>
        </div>
      </div>
    </div>
  );
}

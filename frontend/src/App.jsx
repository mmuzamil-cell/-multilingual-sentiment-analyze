import React, { useState, useEffect } from 'react';
import { 
  Sparkles, 
  Search, 
  UploadCloud, 
  Download, 
  RefreshCw, 
  Trash2, 
  BarChart3, 
  History, 
  Sliders, 
  Info, 
  Home, 
  CheckCircle2, 
  AlertCircle 
} from 'lucide-react';
import { 
  PieChart, 
  Pie, 
  Cell, 
  BarChart as RechartsBarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend, 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  CartesianGrid 
} from 'recharts';

import './style.css';

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 'http://localhost:8000' : window.location.origin + '/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [stats, setStats] = useState({
    total: 0,
    avg_confidence: 0.0,
    positive_ratio: 0.0,
    negative_ratio: 0.0,
    neutral_ratio: 0.0
  });

  // Global fetch stats helper
  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/stats`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (err) {
      console.error("Failed to load statistics:", err);
    }
  };

  useEffect(() => {
    fetchStats();
  }, [activeTab]);

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <Sparkles size={20} />
          <span>Multilingual Sentiment</span>
        </div>
        
        <nav className="sidebar-nav">
          <button 
            className={`nav-item ${activeTab === 'home' ? 'active' : ''}`}
            onClick={() => setActiveTab('home')}
          >
            <Home size={18} />
            <span>Home</span>
          </button>
          
          <button 
            className={`nav-item ${activeTab === 'analyze' ? 'active' : ''}`}
            onClick={() => setActiveTab('analyze')}
          >
            <Sparkles size={18} />
            <span>Analyze Review</span>
          </button>
          
          <button 
            className={`nav-item ${activeTab === 'bulk' ? 'active' : ''}`}
            onClick={() => setActiveTab('bulk')}
          >
            <UploadCloud size={18} />
            <span>Bulk Upload</span>
          </button>
          
          <button 
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <BarChart3 size={18} />
            <span>Analytics Dashboard</span>
          </button>
          
          <button 
            className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <History size={18} />
            <span>History logs</span>
          </button>
          


        </nav>

      </aside>

      {/* Main Page Panel */}
      <main className="content-panel">
        {activeTab === 'home' && <HomeTab setActiveTab={setActiveTab} />}
        {activeTab === 'analyze' && <AnalyzeTab />}
        {activeTab === 'bulk' && <BulkTab />}
        {activeTab === 'dashboard' && <DashboardTab fetchStats={fetchStats} stats={stats} />}
        {activeTab === 'history' && <HistoryTab />}

      </main>
    </div>
  );
}

/* ==========================================
   TAB COMPONENTS
   ========================================== */

// 1. HOME TAB
function HomeTab({ setActiveTab }) {
  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Smart Multilingual Sentiment Analysis</h1>
        <p className="page-subtitle">Understand product reviews across English, Urdu, and Roman Urdu in real time.</p>
      </header>

      <section className="card" style={{ background: 'linear-gradient(to right, #eff6ff, #ffffff)', borderColor: '#bfdbfe' }}>
        <h2 style={{ fontSize: '18px', fontWeight: 600, color: '#1e3a8a', marginBottom: '8px' }}>Deep Learning Powered NLP</h2>

        <button 
          className="btn btn-primary" 
          style={{ marginTop: '16px' }}
          onClick={() => setActiveTab('analyze')}
        >
          Get Started
        </button>
      </section>

      <h2 className="card-title" style={{ marginTop: '32px' }}>Supported Languages</h2>
      <div className="grid-3">
        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '32px', marginBottom: '8px' }}>🇺🇸</div>
          <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '4px' }}>English</h3>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', fontStyle: 'italic', marginBottom: '12px' }}>"This phone is amazing."</p>
        </div>
        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '32px', marginBottom: '8px' }}>🇵🇰</div>
          <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '4px' }}>Urdu</h3>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', fontStyle: 'italic', marginBottom: '12px' }}>"یہ موبائل بہت اچھا ہے۔"</p>
        </div>
        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '32px', marginBottom: '8px' }}>✍️</div>
          <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '4px' }}>Roman Urdu</h3>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', fontStyle: 'italic', marginBottom: '12px' }}>"Ye mobile bohat acha hai."</p>
        </div>
      </div>

      <h2 className="card-title" style={{ marginTop: '32px' }}>Features</h2>
      <div className="grid-3">
        <div className="card">
          <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '8px', color: 'var(--primary)' }}>🧠 Cross-lingual Transformer</h3>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
            Unified semantic mapping across English and Indic scripts. Recognizes context-based sentiments instead of just direct vocabulary matching.
          </p>
        </div>
        <div className="card">
          <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '8px', color: 'var(--primary)' }}>⚡ Keyword Highlighting</h3>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
            Automatically parses input text and color-codes reviews so business users can quickly identify positive and negative sentiment drivers.
          </p>
        </div>
        <div className="card">
          <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '8px', color: 'var(--primary)' }}>📊 Real-time Dashboard</h3>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
            Upload customer review CSV spreadsheets, visualize sentiment patterns over timelines, print PDF summaries, and search past prediction history.
          </p>
        </div>
      </div>
    </div>
  );
}

// 2. ANALYZE REVIEW TAB
function AnalyzeTab() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please enter a review text.");
      return;
    }
    setError('');
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data);
      } else {
        const errData = await res.json();
        setError(errData.detail || "Server error.");
      }
    } catch (err) {
      setError("Failed to connect to the backend server. Make sure uvicorn is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Analyze Review</h1>
        <p className="page-subtitle">Enter a customer review and click analyze to see predicted sentiments and keyword drivers.</p>
      </header>

      <div className="card">
        <div className="form-group">
          <label className="form-label">Review Text</label>
          <textarea 
            className="text-area" 
            rows="5"
            placeholder="Type your product review here in English, Urdu, or Roman Urdu..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          ></textarea>
        </div>

        {error && (
          <div style={{ color: 'var(--danger-text)', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <button 
          className="btn btn-primary" 
          onClick={handleAnalyze}
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Analyze Sentiment"}
        </button>
      </div>

      {loading && <div className="spinner"></div>}

      {result && (
        <div className="card" style={{ borderLeft: `6px solid ${result.sentiment === 'Positive' ? '#2ecc71' : result.sentiment === 'Negative' ? '#e74c3c' : '#f39c12'}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <div>
              <h2 style={{ fontSize: '18px', fontWeight: 600 }}>Classification Results</h2>
              <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Latency: {(result.prediction_time * 1000).toFixed(1)} ms</span>
            </div>
            <span className={`badge ${result.sentiment === 'Positive' ? 'badge-positive' : result.sentiment === 'Negative' ? 'badge-negative' : 'badge-neutral'}`} style={{ fontSize: '14px', padding: '6px 14px' }}>
              {result.sentiment}
            </span>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', display: 'block', marginBottom: '6px' }}>Keyword Highlights</span>
            <div 
              className={`highlight-box ${result.language === 'ur' ? 'urdu-style' : result.language === 'hi' ? 'hindi-style' : ''}`}
              dangerouslySetInnerHTML={{ __html: result.highlighted_text }} 
            />
          </div>

          <div className="grid-3" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '16px', marginTop: '16px' }}>
            <div>
              <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>DETECTED LANGUAGE</span>
              <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                {result.language === 'en' && 'English 🇺🇸'}
                {result.language === 'ur' && 'Urdu 🇵🇰'}
                {result.language === 'ur_roman' && 'Roman Urdu ✍️'}
                {result.language === 'unknown' && 'Unknown ❓'}
              </span>
            </div>
            <div>
              <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>CONFIDENCE</span>
              <span style={{ fontSize: '14px', fontWeight: 600, color: result.sentiment === 'Positive' ? 'var(--success-text)' : result.sentiment === 'Negative' ? 'var(--danger-text)' : 'var(--warning-text)' }}>
                {(result.confidence * 100).toFixed(2)}%
              </span>
            </div>
            <div>
              <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>CLASSIFIER METHOD</span>
              <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                {result.using_ml_model ? "Fine-Tuned XLM-R Model" : "Lexicon Heuristic Fallback"}
              </span>
            </div>
          </div>

          <div style={{ marginTop: '20px' }}>
            <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '6px' }}>Confidence Level</span>
            <div className="progress-bar-container">
              <div 
                className="progress-bar-fill" 
                style={{ 
                  width: `${result.confidence * 100}%`,
                  backgroundColor: result.sentiment === 'Positive' ? '#2ecc71' : result.sentiment === 'Negative' ? '#e74c3c' : '#f39c12'
                }}
              ></div>
            </div>
          </div>
          
          {!result.using_ml_model && (
            <div className="card" style={{ backgroundColor: '#fef3c7', borderColor: '#fcd34d', padding: '12px 16px', marginTop: '20px', marginBottom: 0, display: 'flex', gap: '10px' }}>
              <AlertCircle size={18} style={{ color: '#d97706', flexShrink: 0 }} />
              <p style={{ fontSize: '13px', color: '#b45309', margin: 0 }}>
                <strong>Heuristics Fallback Active:</strong> The local XLM-RoBERTa deep learning model is still downloading or training. Lexicon-based keywords matching is running. The model will take over automatically once training is completed in the background.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// 3. BULK UPLOAD TAB
function BulkTab() {
  const [file, setFile] = useState(null);
  const [columnName, setColumnName] = useState('review');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState('');

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }
    setError('');
    setLoading(true);
    setSummary(null);
    setProgress(20);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("text_column", columnName);

    try {
      setProgress(50);
      const res = await fetch(`${API_BASE}/api/bulk-upload`, {
        method: 'POST',
        body: formData
      });
      
      setProgress(80);
      if (res.ok) {
        const data = await res.json();
        setSummary(data);
        setProgress(100);
      } else {
        const errData = await res.json();
        setError(errData.detail || "Bulk classification failed.");
      }
    } catch (err) {
      setError("Failed to run bulk classification. Verify backend server is alive.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Bulk CSV Upload</h1>
        <p className="page-subtitle">Upload spreadsheets containing text reviews to classify sentiments in batches.</p>
      </header>

      <div className="card">
        <form onSubmit={handleUpload}>
          <div className="form-group">
            <label className="form-label">Select CSV File</label>
            <div className="dropzone" onClick={() => document.getElementById('csv-file-input').click()}>
              <UploadCloud className="dropzone-icon" size={36} style={{ margin: '0 auto 12px' }} />
              <p style={{ fontSize: '14px', fontWeight: 500 }}>
                {file ? file.name : "Drag & drop file or click to browse"}
              </p>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Supports standard CSV files with review columns</span>
              <input 
                type="file" 
                id="csv-file-input" 
                style={{ display: 'none' }} 
                accept=".csv"
                onChange={(e) => setFile(e.target.files[0])}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Review Column Name</label>
            <input 
              type="text" 
              className="text-input" 
              placeholder="e.g. review, comment, text..."
              value={columnName}
              onChange={(e) => setColumnName(e.target.value)}
            />
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Specify the column header title that holds the review sentences</span>
          </div>

          {error && (
            <div style={{ color: 'var(--danger-text)', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          <button 
            type="submit" 
            className="btn btn-primary" 
            disabled={loading}
          >
            {loading ? "Processing..." : "Start Batch Analysis"}
          </button>
        </form>
      </div>

      {loading && (
        <div style={{ marginBottom: '24px' }}>
          <span style={{ fontSize: '13px', display: 'block', marginBottom: '8px', fontWeight: 500 }}>Running batch predictions...</span>
          <div className="progress-bar-container">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
          </div>
        </div>
      )}

      {summary && (
        <div>
          <h2 className="card-title">Analysis Summary</h2>
          <div className="grid-4">
            <div className="metric-card" style={{ borderLeft: '4px solid #2ecc71' }}>
              <div className="metric-label">Positive Reviews</div>
              <div className="metric-value">{summary.positive_count}</div>
              <div className="metric-footer" style={{ color: '#2ecc71' }}>
                {((summary.positive_count / summary.total_rows) * 100).toFixed(1)}% of total
              </div>
            </div>
            
            <div className="metric-card" style={{ borderLeft: '4px solid #f39c12' }}>
              <div className="metric-label">Neutral Reviews</div>
              <div className="metric-value">{summary.neutral_count}</div>
              <div className="metric-footer" style={{ color: '#f39c12' }}>
                {((summary.neutral_count / summary.total_rows) * 100).toFixed(1)}% of total
              </div>
            </div>

            <div className="metric-card" style={{ borderLeft: '4px solid #e74c3c' }}>
              <div className="metric-label">Negative Reviews</div>
              <div className="metric-value">{summary.negative_count}</div>
              <div className="metric-footer" style={{ color: '#e74c3c' }}>
                {((summary.negative_count / summary.total_rows) * 100).toFixed(1)}% of total
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Avg Confidence</div>
              <div className="metric-value">{(summary.avg_confidence * 100).toFixed(1)}%</div>
              <div className="metric-footer" style={{ color: 'var(--primary)' }}>
                Across {summary.total_rows} rows
              </div>
            </div>
          </div>

          <div className="card" style={{ marginTop: '24px' }}>
            <h3 className="card-title">Predictions Logs Preview (Top 10 Rows)</h3>
            <div className="table-wrapper">
              <table className="table-custom">
                <thead>
                  <tr>
                    <th>Review</th>
                    <th>Detected Lang</th>
                    <th>Sentiment</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.preview_records.map((r, i) => (
                    <tr key={i}>
                      <td style={{ maxWidth: '350px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r[columnName]}</td>
                      <td>{(r.detected_language || 'UNKNOWN').toUpperCase()}</td>
                      <td>
                        <span className={`badge ${r.predicted_sentiment === 'Positive' ? 'badge-positive' : r.predicted_sentiment === 'Negative' ? 'badge-negative' : 'badge-neutral'}`} style={{ fontSize: '11px' }}>
                          {r.predicted_sentiment}
                        </span>
                      </td>
                      <td>{((r.confidence_score || 0) * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div style={{ marginTop: '20px', display: 'flex', gap: '12px' }}>
              <a href={`${API_BASE}/api/history/export-csv`} className="btn btn-secondary" style={{ flexGrow: 1, textDecoration: 'none' }}>
                <Download size={16} />
                <span>Export Predictions to CSV</span>
              </a>
              <a href={`${API_BASE}/api/history/export-pdf`} className="btn btn-secondary" style={{ flexGrow: 1, textDecoration: 'none' }}>
                <Download size={16} />
                <span>Download PDF Summary Report</span>
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// 4. ANALYTICS DASHBOARD TAB
function DashboardTab({ fetchStats, stats }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/history`);
      if (res.ok) {
        const history = await res.json();
        setData(history);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadDemo = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/load-demo`, { method: 'POST' });
      if (res.ok) {
        await loadData();
        await fetchStats();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) return <div className="spinner" style={{ marginTop: '100px' }}></div>;

  if (!data || data.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '60px 20px' }}>
        <BarChart3 size={48} style={{ color: 'var(--text-muted)', marginBottom: '16px' }} />
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '8px' }}>No Prediction History Found</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '14px', maxWidth: '450px', margin: '0 auto 24px', lineHeight: '1.5' }}>
          Analyze a single review or upload a CSV containing customer reviews first to populate the analytics dashboard visualizations.
        </p>
        <button className="btn btn-primary" onClick={handleLoadDemo}>
          <RefreshCw size={16} />
          <span>Load Multilingual Demo Data</span>
        </button>
      </div>
    );
  }

  // Compute sentiment distribution
  const sentiments = data.reduce((acc, curr) => {
    acc[curr.prediction] = (acc[curr.prediction] || 0) + 1;
    return acc;
  }, {});

  const pieData = Object.keys(sentiments).map(k => ({
    name: k,
    value: sentiments[k]
  }));

  // Compute language distribution
  const languages = data.reduce((acc, curr) => {
    const codeMap = { en: 'English', ur: 'Urdu', ur_roman: 'Roman Urdu', unknown: 'Unknown' };
    const name = codeMap[curr.language] || curr.language;
    acc[name] = (acc[name] || 0) + 1;
    return acc;
  }, {});

  const barData = Object.keys(languages).map(k => ({
    language: k,
    count: languages[k]
  }));

  // Timeline aggregation
  const timeline = data.reduce((acc, curr) => {
    const date = curr.timestamp.split(' ')[0];
    if (!acc[date]) acc[date] = { date, Positive: 0, Neutral: 0, Negative: 0 };
    acc[date][curr.prediction] = (acc[date][curr.prediction] || 0) + 1;
    return acc;
  }, {});

  const lineData = Object.values(timeline).sort((a, b) => a.date.localeCompare(b.date));

  // Extract top keywords
  // A simple heuristic using lexicon matches or word frequency
  const stopWords = new Set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'it', 'to', 'for', 'of', 'in', 'on', 'this', 'that', 'with', 'ye', 'bohat', 'acha', 'hai', 'ko', 'se', 'me', 'ka', 'ki', 'aur', 'bhi']);
  const wordsMap = {};
  data.forEach(item => {
    const tokens = item.review.toLowerCase().split(/[\s,.!?;:()۔]+/);
    tokens.forEach(t => {
      if (t.length > 3 && !stopWords.has(t) && !t.includes('http')) {
        wordsMap[t] = (wordsMap[t] || 0) + 1;
      }
    });
  });

  const keywords = Object.keys(wordsMap)
    .map(word => ({ word, count: wordsMap[word] }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 15);

  const colors = {
    Positive: '#2ecc71',
    Neutral: '#f39c12',
    Negative: '#e74c3c'
  };

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Analytics Dashboard</h1>
        <p className="page-subtitle">Real-time charts displaying historical predictions and language distributions.</p>
      </header>

      {/* Stats Cards */}
      <div className="grid-4" style={{ marginBottom: '32px' }}>
        <div className="metric-card">
          <div className="metric-label">Total Predictions</div>
          <div className="metric-value">{stats.total || data.length}</div>
        </div>
        <div className="metric-card" style={{ borderLeft: '4px solid #2ecc71' }}>
          <div className="metric-label">Positive Sentiment</div>
          <div className="metric-value">{((stats.positive_ratio || 0.0) * 100).toFixed(1)}%</div>
        </div>
        <div className="metric-card" style={{ borderLeft: '4px solid #e74c3c' }}>
          <div className="metric-label">Negative Sentiment</div>
          <div className="metric-value">{((stats.negative_ratio || 0.0) * 100).toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Model Certainty</div>
          <div className="metric-value">{((stats.avg_confidence || 0.0) * 100).toFixed(1)}%</div>
        </div>
      </div>

      <div className="grid-2">
        {/* Pie Chart */}
        <div className="card" style={{ height: '350px' }}>
          <h3 className="card-title">Sentiment Distribution</h3>
          <div style={{ width: '100%', height: '240px' }}>
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={colors[entry.name] || '#94a3b8'} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend layout="horizontal" align="center" verticalAlign="bottom" />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Bar Chart */}
        <div className="card" style={{ height: '350px' }}>
          <h3 className="card-title">Review Volume by Language</h3>
          <div style={{ width: '100%', height: '240px' }}>
            <ResponsiveContainer>
              <RechartsBarChart data={barData}>
                <XAxis dataKey="language" stroke="#94a3b8" fontSize={12} tickLine={false} />
                <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} />
                <Tooltip cursor={{ fill: 'rgba(0, 0, 0, 0.02)' }} />
                <Bar dataKey="count" fill="var(--primary)" radius={[4, 4, 0, 0]}>
                  {barData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#2563eb' : '#3b82f6'} />
                  ))}
                </Bar>
              </RechartsBarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Timeline Chart */}
      <div className="card" style={{ height: '350px', marginTop: '24px' }}>
        <h3 className="card-title">Sentiment Volume Trends</h3>
        <div style={{ width: '100%', height: '260px' }}>
          <ResponsiveContainer>
            <LineChart data={lineData}>
              <CartesianGrid stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} tickLine={false} />
              <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} />
              <Tooltip />
              <Legend verticalAlign="top" height={36} />
              <Line type="monotone" dataKey="Positive" stroke="#2ecc71" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              <Line type="monotone" dataKey="Neutral" stroke="#f39c12" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              <Line type="monotone" dataKey="Negative" stroke="#e74c3c" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Keywords Badge Cloud */}
      <div className="card" style={{ marginTop: '24px' }}>
        <h3 className="card-title">Top Sentiment Keywords extracted</h3>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>Captured keywords from all query history runs.</p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {keywords.map((kw, i) => (
            <span key={i} className="badge badge-lang" style={{ fontSize: '13px', padding: '6px 12px', display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
              <span>{kw.word}</span>
              <strong style={{ opacity: 0.5 }}>({kw.count})</strong>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// 5. HISTORY TAB
function HistoryTab() {
  const [data, setData] = useState([]);
  const [search, setSearch] = useState('');
  const [sentiment, setSentiment] = useState('All');
  const [language, setLanguage] = useState('All');
  const [loading, setLoading] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const q = new URLSearchParams();
      if (search) q.append('search', search);
      if (sentiment !== 'All') q.append('sentiment', sentiment);
      if (language !== 'All') q.append('language', language);
      
      const res = await fetch(`${API_BASE}/api/history?${q.toString()}`);
      if (res.ok) {
        const history = await res.json();
        setData(history);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history/clear`, { method: 'POST' });
      if (res.ok) {
        setConfirmClear(false);
        fetchHistory();
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [sentiment, language]);

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Prediction History</h1>
        <p className="page-subtitle">Search, query, and download historical review logs.</p>
      </header>

      {/* Filter Options */}
      <div className="card">
        <div className="grid-3">
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Search Query</label>
            <div style={{ position: 'relative' }}>
              <input 
                type="text" 
                className="text-input" 
                placeholder="Search text..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && fetchHistory()}
              />
              <button 
                onClick={fetchHistory}
                style={{ position: 'absolute', right: '10px', top: '10px', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
              >
                <Search size={18} />
              </button>
            </div>
          </div>
          
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Sentiment</label>
            <select 
              className="select-box"
              value={sentiment}
              onChange={(e) => setSentiment(e.target.value)}
            >
              <option value="All">All Sentiments</option>
              <option value="Positive">Positive</option>
              <option value="Neutral">Neutral</option>
              <option value="Negative">Negative</option>
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Language</label>
            <select 
              className="select-box"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
            >
              <option value="All">All Languages</option>
              <option value="English">English</option>
              <option value="Urdu">Urdu</option>
              <option value="Roman Urdu">Roman Urdu</option>
            </select>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="spinner"></div>
      ) : data.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <p style={{ color: 'var(--text-secondary)' }}>No records matching search/filters found.</p>
        </div>
      ) : (
        <div>
          <div className="card">
            <h3 className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Predictions Log ({data.length} records found)</span>
              <div style={{ display: 'flex', gap: '8px' }}>
                <a href={`${API_BASE}/api/history/export-csv`} className="btn btn-secondary" style={{ padding: '6px 12px', textDecoration: 'none' }}>
                  <Download size={14} />
                  <span>CSV</span>
                </a>
                <a href={`${API_BASE}/api/history/export-pdf`} className="btn btn-secondary" style={{ padding: '6px 12px', textDecoration: 'none' }}>
                  <Download size={14} />
                  <span>Executive PDF Report</span>
                </a>
              </div>
            </h3>
            
            <div className="table-wrapper">
              <table className="table-custom">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Review</th>
                    <th>Lang</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((r, i) => (
                    <tr key={i}>
                      <td style={{ whiteSpace: 'nowrap', fontSize: '12px' }}>{r.timestamp}</td>
                      <td style={{ maxWidth: '400px', wordBreak: 'break-word' }}>{r.review}</td>
                      <td style={{ textTransform: 'uppercase' }}>{r.language}</td>
                      <td>
                        <span className={`badge ${r.prediction === 'Positive' ? 'badge-positive' : r.prediction === 'Negative' ? 'badge-negative' : 'badge-neutral'}`} style={{ fontSize: '11px' }}>
                          {r.prediction}
                        </span>
                      </td>
                      <td>{(r.confidence * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Clear Actions */}
          <div className="card" style={{ borderLeft: '4px solid #ef4444', backgroundColor: '#fef2f2' }}>
            <h3 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--danger-text)', marginBottom: '8px' }}>Danger Zone</h3>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>Permanently clear all prediction logs from the SQLite database.</p>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <input 
                type="checkbox" 
                id="confirm-clear-checkbox" 
                checked={confirmClear} 
                onChange={(e) => setConfirmClear(e.target.checked)} 
              />
              <label htmlFor="confirm-clear-checkbox" style={{ fontSize: '13px', color: 'var(--text-secondary)', cursor: 'pointer' }}>I confirm I want to wipe history logs.</label>
              
              <button 
                className="btn btn-danger" 
                onClick={handleClear}
                disabled={!confirmClear}
                style={{ padding: '6px 16px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px', marginLeft: 'auto' }}
              >
                <Trash2 size={14} />
                <span>Delete All History</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


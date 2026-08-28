import { useState } from "react";
import axios from "axios";

const CATEGORY_COLORS = {
  "medical aid": "#e53e3e",
  "shelter": "#3182ce",
  "food/water": "#38a169",
  "rescue/missing": "#d69e2e",
  "other": "#718096",
};

function App() {
  const [text, setText] = useState("");
  const [history, setHistory] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:8000/classify", { text });
      const report = { text, ...response.data, id: Date.now() };
      setHistory([report, ...history]);
      setSelected(report);
      setText("");
    } catch (err) {
      alert("Error: is the backend running on port 8000?");
    }
    setLoading(false);
  };

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "sans-serif" }}>
      {/* Left: report list + input */}
      <div style={{ width: "380px", borderRight: "1px solid #e2e8f0", padding: "20px", overflowY: "auto" }}>
        <h2>Singlish Need Classifier</h2>

        <textarea
          rows="3"
          style={{ width: "100%", padding: "8px", boxSizing: "border-box" }}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type a Singlish need report..."
        />
        <button onClick={handleSubmit} disabled={loading} style={{ marginTop: "8px", width: "100%", padding: "8px" }}>
          {loading ? "Classifying..." : "Classify"}
        </button>

        <h3 style={{ marginTop: "24px" }}>Incoming Reports</h3>
        {history.length === 0 && <p style={{ color: "#888" }}>No reports yet.</p>}
        {history.map((r) => (
          <div
            key={r.id}
            onClick={() => setSelected(r)}
            style={{
              padding: "10px",
              marginBottom: "8px",
              borderRadius: "6px",
              cursor: "pointer",
              background: selected?.id === r.id ? "#edf2f7" : "#fff",
              borderLeft: `4px solid ${CATEGORY_COLORS[r.category] || "#ccc"}`,
              boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
            }}
          >
            <div style={{ fontSize: "13px", fontWeight: "bold", color: CATEGORY_COLORS[r.category] }}>
              {r.category.toUpperCase()} {r.priority ? "🔴" : ""}
            </div>
            <div style={{ fontSize: "13px", color: "#555", marginTop: "4px" }}>
              {r.text.length > 50 ? r.text.slice(0, 50) + "..." : r.text}
            </div>
          </div>
        ))}
      </div>

      {/* Right: detail view */}
      <div style={{ flex: 1, padding: "30px" }}>
        {!selected ? (
          <p style={{ color: "#888" }}>Select a report to see details.</p>
        ) : (
          <div>
            <h2>Report Detail</h2>
            <p><strong>Original text:</strong></p>
            <p style={{ background: "#f7fafc", padding: "12px", borderRadius: "6px" }}>{selected.text}</p>

            <table style={{ marginTop: "20px", borderCollapse: "collapse" }}>
              <tbody>
                <tr><td style={{ padding: "6px 12px", fontWeight: "bold" }}>Category</td><td style={{ padding: "6px 12px" }}>{selected.category}</td></tr>
                <tr><td style={{ padding: "6px 12px", fontWeight: "bold" }}>Confidence</td><td style={{ padding: "6px 12px" }}>{(selected.confidence * 100).toFixed(1)}%</td></tr>
                <tr><td style={{ padding: "6px 12px", fontWeight: "bold" }}>Responder</td><td style={{ padding: "6px 12px" }}>{selected.responder}</td></tr>
                <tr><td style={{ padding: "6px 12px", fontWeight: "bold" }}>Priority</td><td style={{ padding: "6px 12px" }}>{selected.priority ? "🔴 Urgent" : "Normal"}</td></tr>
              </tbody>
            </table>

            <h3 style={{ marginTop: "24px" }}>All Category Scores</h3>
            {Object.entries(selected.all_scores).map(([cat, score]) => (
              <div key={cat} style={{ marginBottom: "6px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
                  <span>{cat}</span><span>{(score * 100).toFixed(1)}%</span>
                </div>
                <div style={{ background: "#e2e8f0", borderRadius: "4px", height: "6px" }}>
                  <div style={{ width: `${score * 100}%`, background: CATEGORY_COLORS[cat], height: "6px", borderRadius: "4px" }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
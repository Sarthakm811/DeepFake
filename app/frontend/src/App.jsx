import { useMemo, useState } from "react";
import axios from "axios";
import {
    Bar,
    BarChart,
    CartesianGrid,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from "recharts";

const initialResult = null;

export default function App() {
    const [ mode, setMode ] = useState("text");
    const [ text, setText ] = useState("");
    const [ image, setImage ] = useState(null);
    const [ video, setVideo ] = useState(null);
    const [ loading, setLoading ] = useState(false);
    const [ result, setResult ] = useState(initialResult);
    const [ error, setError ] = useState("");

    const modalityReady = mode === "text"
        ? Number(Boolean(text.trim()))
        : mode === "image"
            ? Number(Boolean(image))
            : Number(Boolean(video));

    const chartData = useMemo(() => {
        if (!result) return [];
        return [
            { modality: "Text", score: result.scores.text ?? 0.5 },
            { modality: "Image", score: result.scores.image ?? 0.5 },
            { modality: "Video", score: result.scores.video ?? 0.5 }
        ];
    }, [ result ]);

    const handleAnalyze = async () => {
        if (mode === "text" && !text.trim()) {
            setError("Please provide text input.");
            return;
        }
        if (mode === "image" && !image) {
            setError("Please upload an image.");
            return;
        }
        if (mode === "video" && !video) {
            setError("Please upload a video.");
            return;
        }

        setLoading(true);
        setError("");
        setResult(initialResult);

        const form = new FormData();
        if (mode === "text") form.append("text", text.trim());
        if (mode === "image" && image) form.append("image", image);
        if (mode === "video" && video) form.append("video", video);

        try {
            const response = await axios.post("/api/analyze", form, {
                headers: { "Content-Type": "multipart/form-data" }
            });
            setResult(response.data);
        } catch (e) {
            setError(e?.response?.data?.detail || e.message || "Failed to analyze input.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="page">
            <header className="hero">
                <h1>🕵️ Deepfake Detection Studio</h1>
                <p>React-powered dashboard for multi-modal deepfake analysis</p>
            </header>

            <main className="grid">
                <section className="card input-card">
                    <h2>Inputs</h2>
                    <div className="mode-row">
                        <label className="mode-option">
                            <input
                                type="radio"
                                name="mode"
                                checked={mode === "text"}
                                onChange={() => setMode("text")}
                            />
                            Text
                        </label>
                        <label className="mode-option">
                            <input
                                type="radio"
                                name="mode"
                                checked={mode === "image"}
                                onChange={() => setMode("image")}
                            />
                            Image
                        </label>
                        <label className="mode-option">
                            <input
                                type="radio"
                                name="mode"
                                checked={mode === "video"}
                                onChange={() => setMode("video")}
                            />
                            Video
                        </label>
                    </div>

                    <label>
                        Text
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            placeholder="Paste suspicious text or caption..."
                            rows={5}
                            disabled={mode !== "text"}
                        />
                    </label>

                    <div className="file-row">
                        <label>
                            Image
                            <input
                                type="file"
                                accept="image/*"
                                disabled={mode !== "image"}
                                onChange={(e) => setImage(e.target.files?.[ 0 ] || null)}
                            />
                        </label>
                        <label>
                            Video
                            <input
                                type="file"
                                accept="video/*"
                                disabled={mode !== "video"}
                                onChange={(e) => setVideo(e.target.files?.[ 0 ] || null)}
                            />
                        </label>
                    </div>

                    <div className="status-row">
                        <span>Selected modality: {mode.toUpperCase()}</span>
                        <span>Input ready: {modalityReady ? "Yes" : "No"}</span>
                        <progress value={modalityReady} max={1} />
                    </div>

                    <button className="primary-btn" onClick={handleAnalyze} disabled={loading}>
                        {loading ? "Analyzing..." : "Analyze Deepfake"}
                    </button>

                    {error && <p className="error">{error}</p>}
                </section>

                <section className="card output-card">
                    <h2>Results</h2>
                    {!result ? (
                        <p className="muted">Run analysis to view model output.</p>
                    ) : (
                        <>
                            <div className="score-hero">
                                <span className="score-label">Final Probability</span>
                                <strong>{(result.final_score * 100).toFixed(1)}%</strong>
                                <span className={result.label === "FAKE" ? "chip danger" : "chip safe"}>{result.label}</span>
                                <span className="chip">Mode: {(result.selected_modality || mode).toUpperCase()}</span>
                            </div>

                            <div className="modality-grid">
                                <ModalityCard title="Text" score={result.scores.text} state={result.states.text} />
                                <ModalityCard title="Image" score={result.scores.image} state={result.states.image} />
                                <ModalityCard title="Video" score={result.scores.video} state={result.states.video} />
                            </div>

                            <div className="chart-wrap">
                                <ResponsiveContainer width="100%" height={240}>
                                    <BarChart data={chartData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                        <XAxis dataKey="modality" stroke="#94a3b8" />
                                        <YAxis domain={[ 0, 1 ]} stroke="#94a3b8" />
                                        <Tooltip />
                                        <Bar dataKey="score" fill="#60a5fa" radius={[ 8, 8, 0, 0 ]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            <details>
                                <summary>Raw JSON</summary>
                                <pre>{JSON.stringify(result, null, 2)}</pre>
                            </details>
                        </>
                    )}
                </section>
            </main>
        </div>
    );
}

function ModalityCard({ title, score, state }) {
    return (
        <article className="modality-card">
            <h3>{title}</h3>
            <p className="modality-score">{score == null ? "N/A" : `${(score * 100).toFixed(1)}%`}</p>
            <p className="muted">{state}</p>
        </article>
    );
}

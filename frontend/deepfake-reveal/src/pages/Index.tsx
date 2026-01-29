import { useState } from "react";
import { Shield, FileText, Upload, Type } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { UploadZone } from "@/components/UploadZone";
import { VideoPreview } from "@/components/VideoPreview";
import { AnalysisResult } from "@/components/AnalysisResult";
import { TextAnalysisResult } from "@/components/TextAnalysisResult";
import { AnalyzingState } from "@/components/AnalyzingState";
import { toast } from "sonner";
import heroBackground from "@/assets/hero-background.jpg";

const Index = () => {
  const [activeTab, setActiveTab] = useState<"video" | "text">("video");

  // Video State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  // Text State
  const [textInput, setTextInput] = useState("");

  // Common State
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Results
  const [videoResult, setVideoResult] = useState<{
    prediction: "real" | "fake";
    confidence: number;
  } | null>(null);

  const [textResult, setTextResult] = useState<{
    label: "Human" | "AI";
    confidence: number;
    aiProbability?: number;
  } | null>(null);

  const API_URL =
    import.meta.env.VITE_API_URL ||
    "http://localhost:8000";
  "https://shreesha1-deepfake.hf.space";

  // --- Handlers ---

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setVideoResult(null);

    // Create object URL for video preview
    const url = URL.createObjectURL(file);
    setVideoUrl(url);

    toast.success("File loaded successfully");
  };

  const handleVideoAnalyze = async () => {
    if (!selectedFile) {
      toast.error("Please upload a file first");
      return;
    }

    setIsAnalyzing(true);
    setVideoResult(null);

    const endpoint = selectedFile.type.startsWith("image/") ? "/analyze-image" : "/predict";

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Backend error");

      const data = await response.json();
      console.log("API response:", data);

      // Normalize response
      // Image endpoint returns: is_fake (bool), prediction (str "FAKE"/"REAL")
      // Video endpoint returns: prediction (str "FAKE"/"REAL"), is_fake (bool)

      const isFake = data.is_fake;

      setVideoResult({
        prediction: isFake ? "fake" : "real",
        confidence: data.confidence ?? 0,
      });

      toast.success("Analysis complete!");
    } catch (err) {
      console.error(err);
      toast.error("Analysis failed. Is the backend running?");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTextAnalyze = async () => {
    if (!textInput.trim()) {
      toast.error("Please enter some text");
      return;
    }

    setIsAnalyzing(true);
    setTextResult(null);

    try {
      const response = await fetch(`${API_URL}/analyze-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput.trim() })
      });

      if (!response.ok) throw new Error("Backend error");

      const data = await response.json();

      setTextResult({
        label: data.label === "AI" ? "AI" : "Human",
        confidence: data.confidence,
        aiProbability: data.ai_probability
      });

      toast.success("Text analysis complete!");
    } catch (err) {
      console.error(err);
      toast.error("Text analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    if (activeTab === "video") {
      setSelectedFile(null);
      setVideoUrl(null);
      setVideoResult(null);
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    } else {
      setTextInput("");
      setTextResult(null);
    }
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div
        className="relative min-h-screen flex flex-col"
        style={{
          backgroundImage: `url(${heroBackground})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundAttachment: 'fixed',
        }}
      >
        {/* Overlay */}
        <div className="absolute inset-0 bg-background/90 backdrop-blur-sm" />

        {/* Content */}
        <div className="relative z-10 flex-1">
          {/* Header */}
          <header className="border-b border-border/50 backdrop-blur-md bg-background/50">
            <div className="container mx-auto px-6 py-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10 border border-primary/30">
                  <Shield className="w-6 h-6 text-primary" />
                </div>
                <h1 className="text-2xl font-bold font-display text-foreground">
                  SafeGuard AI
                </h1>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main className="container mx-auto px-6 py-12">
            <div className="max-w-7xl mx-auto">

              {/* Title */}
              <div className="text-center mb-12 space-y-4">
                <h2 className="text-5xl md:text-6xl font-bold font-display text-foreground leading-tight">
                  AI Content Detection
                  <span className="block text-primary mt-2">Deepfake + Text Analysis</span>
                </h2>
                <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                  Detect AI-Content videos, images, and text with advanced neural networks.
                </p>
              </div>

              {/* Service Tabs */}
              <div className="flex justify-center mb-12">
                <div className="inline-flex rounded-xl border border-border bg-card/50 p-1.5 gap-2 shadow-lg backdrop-blur-sm">
                  <button
                    onClick={() => { setActiveTab("video"); setVideoResult(null); }}
                    className={`flex items-center gap-2 px-6 py-3 text-sm font-medium rounded-lg transition-all duration-300 ${activeTab === "video"
                      ? "bg-primary text-primary-foreground shadow-md scale-105"
                      : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                      }`}
                  >
                    <Upload className="w-4 h-4" />
                    Deepfake Detector
                  </button>
                  <button
                    onClick={() => { setActiveTab("text"); setTextResult(null); }}
                    className={`flex items-center gap-2 px-6 py-3 text-sm font-medium rounded-lg transition-all duration-300 ${activeTab === "text"
                      ? "bg-primary text-primary-foreground shadow-md scale-105"
                      : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                      }`}
                  >
                    <Type className="w-4 h-4" />
                    AI Text Detector
                  </button>
                </div>
              </div>

              {/* VIDEO ANALYSIS UI */}
              {activeTab === "video" && (
                <div className="animate-fade-in space-y-8">
                  <div className="grid lg:grid-cols-2 gap-8">
                    {/* Left: Upload */}
                    <div className="space-y-6">
                      <div className="bg-card/80 backdrop-blur-sm rounded-2xl border border-border p-6 shadow-xl">
                        <h3 className="text-xl font-semibold font-display mb-4 text-foreground flex items-center gap-2">
                          <Upload className="w-5 h-5 text-primary" />
                          Upload Media
                        </h3>
                        <div className="text-sm text-muted-foreground mb-4">
                          Supports Videos (.mp4, .mov, .avi) and Images (.jpg, .png, .webp).
                        </div>
                        <UploadZone
                          onFileSelect={handleFileSelect}
                          isAnalyzing={isAnalyzing}
                        />
                        {selectedFile && (
                          <div className="mt-4 flex items-center justify-between p-4 bg-muted/30 rounded-lg border border-border/50">
                            <div className="flex items-center gap-3">
                              <div className="p-2 rounded bg-primary/10">
                                <Shield className="w-4 h-4 text-primary" />
                              </div>
                              <div>
                                <p className="text-sm font-medium text-foreground">{selectedFile.name}</p>
                                <p className="text-xs text-muted-foreground">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                              </div>
                            </div>
                            <Button variant="ghost" size="sm" onClick={handleReset} disabled={isAnalyzing}>Remove</Button>
                          </div>
                        )}
                      </div>

                      <Button
                        onClick={handleVideoAnalyze}
                        disabled={!selectedFile || isAnalyzing}
                        className="w-full h-14 text-lg font-semibold bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg hover:shadow-xl transition-all duration-300 rounded-xl"
                      >
                        {isAnalyzing ? "Analyzing Media..." : "Analyze Media"}
                      </Button>
                    </div>

                    {/* Right: Preview & Result */}
                    <div className="space-y-6">
                      {/* Preview */}
                      <div className="bg-card/80 backdrop-blur-sm rounded-2xl border border-border p-6 shadow-xl">
                        <h3 className="text-xl font-semibold font-display mb-4 text-foreground">Preview</h3>
                        <div className="aspect-video bg-black/5 rounded-lg overflow-hidden flex items-center justify-center">
                          {videoUrl ? (
                            selectedFile?.type.startsWith('image') ? (
                              <img src={videoUrl} alt="Preview" className="w-full h-full object-contain" />
                            ) : (
                              <VideoPreview videoUrl={videoUrl} />
                            )
                          ) : (
                            <div className="text-muted-foreground text-sm flex flex-col items-center gap-2">
                              <Shield className="w-8 h-8 opacity-20" />
                              No media selected
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Result */}
                      {(isAnalyzing || videoResult) && (
                        <div>
                          {isAnalyzing ? (
                            <AnalyzingState />
                          ) : videoResult ? (
                            <AnalysisResult
                              prediction={videoResult.prediction}
                              confidence={videoResult.confidence}
                            />
                          ) : null}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* TEXT ANALYSIS UI */}
              {activeTab === "text" && (
                <div className="animate-fade-in max-w-4xl mx-auto space-y-8">
                  <div className="bg-card/80 backdrop-blur-sm rounded-2xl border border-border p-6 shadow-xl">
                    <h3 className="text-xl font-semibold font-display mb-4 text-foreground flex items-center gap-2">
                      <FileText className="w-5 h-5 text-primary" />
                      Analyze Text
                    </h3>
                    <Textarea
                      placeholder="Paste text here to detect if it was written by AI..."
                      value={textInput}
                      onChange={(e) => setTextInput(e.target.value)}
                      className="min-h-[200px] text-lg p-4 bg-muted/30 resize-y rounded-xl border-border/50 focus:border-primary/50 transition-all"
                      disabled={isAnalyzing}
                    />
                    <div className="mt-4 flex justify-between items-center text-sm text-muted-foreground">
                      <span>{textInput.split(/\s+/).filter(Boolean).length} words</span>
                      {textInput && (
                        <Button variant="ghost" size="sm" onClick={() => setTextInput("")} disabled={isAnalyzing}>Clear</Button>
                      )}
                    </div>
                  </div>

                  <div className="flex gap-4">
                    <Button
                      onClick={handleTextAnalyze}
                      disabled={!textInput.trim() || isAnalyzing}
                      className="flex-1 h-14 text-lg font-semibold bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg hover:shadow-xl transition-all duration-300 rounded-xl"
                    >
                      {isAnalyzing ? "Analyzing Text..." : "Detect AI Content"}
                    </Button>
                  </div>

                  {/* Text Result */}
                  {(isAnalyzing || textResult) && (
                    <div>
                      {isAnalyzing ? (
                        <AnalyzingState />
                      ) : textResult ? (
                        <TextAnalysisResult
                          label={textResult.label}
                          confidence={textResult.confidence}
                          aiProbability={textResult.aiProbability}
                        />
                      ) : null}
                    </div>
                  )}
                </div>
              )}

            </div>
          </main>

          {/* Footer */}
          <footer className="border-t border-border/50 backdrop-blur-md bg-background/50 mt-auto">
            <div className="container mx-auto px-6 py-6">
              <p className="text-center text-sm text-muted-foreground">
                Built with DeepGuard AI • Advanced Deepfake Detection
              </p>
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
};

export default Index;

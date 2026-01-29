import { Shield, AlertTriangle, FileText, AlertCircle, CheckCircle } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface TextAnalysis {
  status: string;
  overall_label: string;
  overall_confidence: number;
  ai_probability: number;
  paragraph_count: number;
  ai_paragraph_count: number;
}

interface CombinedVerdict {
  verdict: string;
  severity: "low" | "medium" | "high" | "unknown";
  explanation: string;
}

interface VideoAnalysis {
  status: string;
  prediction: string | null;
  confidence: number | null;
  is_fake: boolean | null;
  message?: string;
}

interface CombinedAnalysisResultProps {
  videoAnalysis: VideoAnalysis;
  textAnalysis: TextAnalysis | null;
  combinedVerdict: CombinedVerdict;
}

const getSeverityColors = (severity: string) => {
  switch (severity) {
    case "high":
      return {
        bg: "bg-destructive/10",
        border: "border-destructive",
        text: "text-destructive",
        glow: "glow-fake"
      };
    case "medium":
      return {
        bg: "bg-yellow-500/10",
        border: "border-yellow-500",
        text: "text-yellow-500",
        glow: ""
      };
    case "low":
      return {
        bg: "bg-primary/10",
        border: "border-primary",
        text: "text-primary",
        glow: "glow-real"
      };
    default:
      return {
        bg: "bg-muted/10",
        border: "border-muted-foreground",
        text: "text-muted-foreground",
        glow: ""
      };
  }
};

const getVerdictIcon = (severity: string) => {
  switch (severity) {
    case "high":
      return <AlertTriangle className="w-10 h-10 text-destructive" />;
    case "medium":
      return <AlertCircle className="w-10 h-10 text-yellow-500" />;
    case "low":
      return <CheckCircle className="w-10 h-10 text-primary" />;
    default:
      return <Shield className="w-10 h-10 text-muted-foreground" />;
  }
};

const getVerdictTitle = (verdict: string) => {
  switch (verdict) {
    case "HIGH_RISK_DEEPFAKE":
      return "HIGH RISK - DEEPFAKE + AI TEXT";
    case "DEEPFAKE_DETECTED":
      return "DEEPFAKE DETECTED";
    case "SUSPICIOUS_CONTEXT":
      return "SUSPICIOUS CONTEXT";
    case "LIKELY_AUTHENTIC":
      return "LIKELY AUTHENTIC";
    case "INCONCLUSIVE":
      return "INCONCLUSIVE";
    default:
      return verdict;
  }
};

export const CombinedAnalysisResult = ({ 
  videoAnalysis, 
  textAnalysis, 
  combinedVerdict 
}: CombinedAnalysisResultProps) => {
  const colors = getSeverityColors(combinedVerdict.severity);

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Combined Verdict */}
      <div className={cn(
        "rounded-2xl p-8 border-2 transition-all duration-500",
        colors.bg, colors.border, colors.glow
      )}>
        <div className="flex items-center gap-4 mb-6">
          <div className={cn(
            "p-4 rounded-full animate-pulse-glow",
            combinedVerdict.severity === "high" ? "bg-destructive/20" : 
            combinedVerdict.severity === "medium" ? "bg-yellow-500/20" : "bg-primary/20"
          )}>
            {getVerdictIcon(combinedVerdict.severity)}
          </div>
          
          <div className="flex-1">
            <h3 className={cn(
              "text-2xl font-bold font-display mb-1",
              colors.text
            )}>
              {getVerdictTitle(combinedVerdict.verdict)}
            </h3>
            <p className="text-muted-foreground text-sm">
              Combined Analysis Result
            </p>
          </div>
        </div>

        <div className={cn(
          "p-4 rounded-lg border text-sm",
          colors.bg, `${colors.border}/30`
        )}>
          <p className="text-foreground/90">{combinedVerdict.explanation}</p>
        </div>
      </div>

      {/* Detailed Results Grid */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Video Analysis Card */}
        <div className="bg-card/80 backdrop-blur-sm rounded-xl border border-border p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-primary/10">
              <Shield className="w-5 h-5 text-primary" />
            </div>
            <h4 className="font-semibold text-foreground">Video Analysis</h4>
          </div>
          
          {videoAnalysis.status === "success" && videoAnalysis.prediction ? (
            <>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Prediction</span>
                  <span className={cn(
                    "font-bold",
                    videoAnalysis.is_fake ? "text-destructive" : "text-primary"
                  )}>
                    {videoAnalysis.prediction}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Confidence</span>
                  <span className="font-bold text-foreground">
                    {videoAnalysis.confidence?.toFixed(1)}%
                  </span>
                </div>
                <Progress 
                  value={videoAnalysis.confidence || 0} 
                  className={cn(
                    "h-2",
                    videoAnalysis.is_fake ? "[&>div]:bg-destructive" : "[&>div]:bg-primary"
                  )}
                />
              </div>
            </>
          ) : (
            <div className="text-center py-4">
              <p className="text-sm text-muted-foreground">
                {videoAnalysis.message || "Could not analyze video"}
              </p>
            </div>
          )}
        </div>

        {/* Text Analysis Card */}
        <div className="bg-card/80 backdrop-blur-sm rounded-xl border border-border p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-yellow-500/10">
              <FileText className="w-5 h-5 text-yellow-500" />
            </div>
            <h4 className="font-semibold text-foreground">Text Context Analysis</h4>
          </div>
          
          {textAnalysis && textAnalysis.status === "success" ? (
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Classification</span>
                <span className={cn(
                  "font-bold",
                  textAnalysis.overall_label === "AI" ? "text-yellow-500" : "text-primary"
                )}>
                  {textAnalysis.overall_label}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Confidence</span>
                <span className="font-bold text-foreground">
                  {textAnalysis.overall_confidence?.toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">AI Paragraphs</span>
                <span className="font-medium text-foreground">
                  {textAnalysis.ai_paragraph_count} / {textAnalysis.paragraph_count}
                </span>
              </div>
              <Progress 
                value={textAnalysis.ai_probability} 
                className={cn(
                  "h-2",
                  textAnalysis.overall_label === "AI" ? "[&>div]:bg-yellow-500" : "[&>div]:bg-primary"
                )}
              />
            </div>
          ) : (
            <div className="text-center py-4">
              <p className="text-sm text-muted-foreground">
                {textAnalysis?.status === "error" 
                  ? "Text analysis unavailable" 
                  : "No text context provided"}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

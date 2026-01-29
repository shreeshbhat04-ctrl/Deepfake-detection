import { FileText, Bot, AlertTriangle, CheckCircle } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface TextAnalysisResultProps {
    label: "Human" | "AI";
    confidence: number;
    aiProbability?: number;
}

export const TextAnalysisResult = ({ label, confidence, aiProbability }: TextAnalysisResultProps) => {
    const isHuman = label === "Human";

    return (
        <div className={cn(
            "animate-slide-up rounded-2xl p-8 border-2 transition-all duration-500",
            isHuman
                ? "bg-green-500/10 border-green-500 glow-real"
                : "bg-purple-500/10 border-purple-500 glow-fake"
        )}>
            <div className="flex items-center gap-4 mb-6">
                <div className={cn(
                    "p-4 rounded-full animate-pulse-glow",
                    isHuman ? "bg-green-500/20" : "bg-purple-500/20"
                )}>
                    {isHuman ? (
                        <CheckCircle className={cn("w-10 h-10", "text-green-500")} />
                    ) : (
                        <Bot className={cn("w-10 h-10", "text-purple-500")} />
                    )}
                </div>

                <div className="flex-1">
                    <h3 className={cn(
                        "text-3xl font-bold font-display mb-1",
                        isHuman ? "text-green-500" : "text-purple-500"
                    )}>
                        {isHuman ? "HUMAN WRITTEN" : "AI GENERATED"}
                    </h3>
                    <p className="text-muted-foreground">
                        {isHuman
                            ? "This text appears to be written by a human."
                            : "This text likely contains AI-generated content."}
                    </p>
                </div>
            </div>

            <div className="space-y-3">
                <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-foreground">Confidence Level</span>
                    <span className={cn(
                        "text-2xl font-bold font-display",
                        isHuman ? "text-green-500" : "text-purple-500"
                    )}>
                        {confidence.toFixed(1)}%
                    </span>
                </div>

                <Progress
                    value={confidence}
                    className={cn(
                        "h-3",
                        isHuman ? "[&>div]:bg-green-500" : "[&>div]:bg-purple-500"
                    )}
                />

                {aiProbability !== undefined && (
                    <div className="flex justify-between items-center mt-2 text-xs text-muted-foreground">
                        <span>AI Probability Score:</span>
                        <span>{(aiProbability * 100).toFixed(1)}%</span>
                    </div>
                )}
            </div>
        </div>
    );
};

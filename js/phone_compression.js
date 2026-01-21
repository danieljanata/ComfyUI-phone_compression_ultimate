// PhoneCompressionUltimate - Custom Slider UI
import { app } from "../../scripts/app.js";

// Register extension for better slider display
app.registerExtension({
    name: "PhoneCompressionUltimate.Sliders",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "PhoneCompressionUltimate") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                
                // Style the node
                this.color = "#2a363b";
                this.bgcolor = "#1a2327";
                
                // Add visual indicator for auto-brightness
                const origDrawForeground = this.onDrawForeground;
                this.onDrawForeground = function(ctx) {
                    if (origDrawForeground) origDrawForeground.apply(this, [ctx]);
                    
                    // Draw phone icon
                    if (!this.flags.collapsed) {
                        ctx.save();
                        ctx.fillStyle = "#ffffff40";
                        ctx.font = "16px Arial";
                        ctx.fillText("📱", this.size[0] - 30, 20);
                        ctx.restore();
                    }
                };
            };
        }
    }
});

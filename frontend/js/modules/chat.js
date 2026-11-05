/**
 * chat.js - Immersive AI Doctor Chat Module
 */

import state from './state.js';
import API from './api.js';

const CHAT = {
    processAIContent(text) {
        // Replace terminal-style progress bars [######] 80%
        return text.replace(/\[([#=->\s]{5,})\]\s*(\d+)(?:\s*%)?/g, (match, bars, percent) => {
            const p = parseInt(percent);
            let colorClass = 'success';
            if (p > 70) colorClass = 'danger';
            else if (p > 40) colorClass = 'warning';
            
            return `
                <div class="ai-progress-wrapper">
                    <div class="ai-progress-label">
                        <span>Độ nguy hiểm:</span>
                        <span>${p}%</span>
                    </div>
                    <div class="ai-progress-container">
                        <div class="ai-progress-fill ${colorClass}" style="width: ${p}%"></div>
                    </div>
                </div>`;
        });
    },

    async sendMessage(chatInput, chatHistory, btnSendMessage, chatModelSelect) {
        const text = chatInput.value.trim();
        if (!text && !state.isGeneratingResponse) return;
        if (state.isGeneratingResponse) return;

        // UI Updates
        const chatWelcome = document.getElementById('chatWelcome');
        if (chatWelcome) {
            chatWelcome.style.opacity = '0';
            setTimeout(() => { chatWelcome.style.display = 'none'; }, 500);
        }

        // Add User Bubble
        const userWrapper = document.createElement('div');
        userWrapper.className = 'message-wrapper';
        userWrapper.innerHTML = `
            <div class="message-bubble-modern msg-user-modern">
                ${text.replace(/\n/g, '<br>')}
            </div>
        `;
        chatHistory.appendChild(userWrapper);
        
        chatInput.value = '';
        chatInput.style.height = '44px';
        chatHistory.scrollTop = chatHistory.scrollHeight;

        // Change Send button to Stop
        btnSendMessage.innerHTML = '<i class="fa-solid fa-stop"></i>';
        btnSendMessage.style.background = 'var(--text-primary)';
        state.isGeneratingResponse = true;

        // AI Placeholder
        const aiWrapper = document.createElement('div');
        aiWrapper.className = 'message-wrapper';
        aiWrapper.innerHTML = `
            <div style="display: flex; gap: 12px; max-width: 100%;">
                <div class="message-avatar avatar-bot"><img src="assets/img/logo.png" alt="AI"></div>
                <div class="message-bubble-modern msg-bot-modern" style="flex: 1;">
                    <div class="ai-status-msg-container">
                        <div class="ai-status-dot"></div>
                        <div class="ai-status-text">Đang kết nối với Skinderm AI...</div>
                    </div>
                    <div class="thinking-status-video active">
                        <video src="assets/video/logo.mp4" autoplay loop muted playsinline></video>
                    </div>
                </div>
            </div>
        `;
        chatHistory.appendChild(aiWrapper);
        
        const aiBubble = aiWrapper.querySelector('.msg-bot-modern');
        const thinkingVideo = aiWrapper.querySelector('.thinking-status-video');
        const statusContainer = aiWrapper.querySelector('.ai-status-msg-container');
        const statusText = aiWrapper.querySelector('.ai-status-text');
        
        const updateStatus = (msg) => { if (statusText) statusText.textContent = msg; };
        chatHistory.scrollTop = chatHistory.scrollHeight;

        state.abortController = new AbortController();

        try {
            updateStatus("Đang đọc dữ liệu chẩn đoán của bạn...");
            
            let context = "Chưa có dữ liệu scan.";
            if (state.history.length > 0) {
                const latest = state.history[0];
                const details = latest.details || {};
                
                // Get weather fields safely from details or current weather state
                const uv = details.uv_index !== undefined && details.uv_index !== null ? details.uv_index : (state.currentWeather?.uvIndex || "Chưa có dữ liệu");
                const temp = details.temperature !== undefined && details.temperature !== null ? details.temperature : (state.currentWeather?.temp || "Chưa có dữ liệu");
                const loc = details.location || (state.currentWeather?.location || "Chưa có dữ liệu");
                
                // Format top 3 predictions
                const top3Str = details.top3 ? details.top3.map(p => `- ${p.label}: ${p.score}%`).join("\n") : "- Không có dữ liệu top 3";
                
                context = `
=== THÔNG TIN BỆNH NHÂN & LỊCH SỬ CHẨN ĐOÁN CHI TIẾT ===
- Chẩn đoán chính: ${latest.diagnosis || "Chưa xác định"} (Độ tin cậy: ${details.confidence || 0}%)
- Mức độ rủi ro tổng hợp: ${latest.risk || 0}%
- Top 3 nguy cơ hàng đầu:
${top3Str}
- Phân tích chuẩn ABCDE:
  + A (Asymmetry - Bất đối xứng): ${details.abcde?.A_asymmetry?.score || 0}% - Trạng thái: ${details.abcde?.A_asymmetry?.status || "Bình thường"}
  + B (Border - Đường viền): ${details.abcde?.B_border?.score || 0}% - Trạng thái: ${details.abcde?.B_border?.status || "Bình thường"}
  + C (Color - Màu sắc): ${details.abcde?.C_color?.score || 0}% - Trạng thái: ${details.abcde?.C_color?.status || "Đều màu"}
  + D (Diameter - Đường kính): ${details.abcde?.D_diameter?.score || 0}% - Chiều dài: ${details.abcde?.D_diameter?.value || 3.0}mm
  + E (Evolution - Tiến triển): ${details.abcde?.E_evolution?.score || 0}% - Trạng thái: ${details.abcde?.E_evolution?.status || "Cần lịch sử"}
- Tình trạng thời tiết & môi trường xung quanh:
  + Chỉ số tia cực tím (UV Index): ${uv}
  + Nhiệt độ môi trường: ${temp}°C
  + Vị trí địa lý: ${loc}
- Lời khuyên ban đầu từ mô hình AI: "${details.medical_advice || "Không có"}"
======================================================
`;
            }
            
            updateStatus("Đang phân tích yêu cầu...");
            const response = await API.startChatStream(text, chatModelSelect.value, context, state.abortController.signal);

            if (!response.ok) throw new Error("Lỗi kết nối máy chủ AI");

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            
            let fullAiRes = "";
            let thoughtElement = null;
            let contentElement = null;
            let sseBuffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                sseBuffer += decoder.decode(value, { stream: true });
                const lines = sseBuffer.split('\n');
                sseBuffer = lines.pop() || "";
                
                for (const rawLine of lines) {
                    const line = rawLine.trim();
                    if (line.startsWith('data: ') && !line.includes('[DONE]')) {
                        let data;
                        try {
                            data = JSON.parse(line.slice(6));
                        } catch (e) {
                            console.error("Stream parse error:", e);
                            continue;
                        }

                        if (data.error) {
                            throw new Error(data.error);
                        }
                        
                        const deltaObj = data.choices?.[0]?.delta || {};
                        const delta = deltaObj.content || "";
                        if (delta) {
                            fullAiRes += delta;

                            // Extract Thought process
                            const thoughtMatch = fullAiRes.match(/(?:\(think\)|<think>|<thought>)([\s\S]*?)(?:\(\/think\)|<\/think>|<\/thought>|$)/);
                            if (thoughtMatch) {
                                if (!thoughtElement) {
                                    updateStatus("AI đang suy nghĩ...");
                                    thoughtElement = document.createElement('div');
                                    thoughtElement.className = 'msg-thinking-container';
                                    if (!state.isThinkingEnabled) thoughtElement.classList.add('hidden-thinking');
                                    
                                    thoughtElement.innerHTML = `<div class="msg-thinking-header"><i class="fa-solid fa-brain"></i> Quá trình suy nghĩ <i class="fa-solid fa-chevron-down" style="margin-left: auto; font-size: 0.7rem;"></i></div><div class="msg-thinking-content"></div>`;
                                    
                                    if(contentElement) aiBubble.insertBefore(thoughtElement, contentElement);
                                    else aiBubble.appendChild(thoughtElement);

                                    thoughtElement.querySelector('.msg-thinking-header').addEventListener('click', () => {
                                        const content = thoughtElement.querySelector('.msg-thinking-content');
                                        content.style.display = content.style.display === 'none' ? 'block' : 'none';
                                    });
                                }
                                thoughtElement.querySelector('.msg-thinking-content').innerHTML = marked.parse(thoughtMatch[1]);
                            }

                            // Main Content
                            let displayContent = fullAiRes.replace(/(?:\(think\)|<think>|<thought>)[\s\S]*?(?:\(\/think\)|<\/think>|<\/thought>|$)/g, '').trim();
                            if (displayContent) {
                                if(thinkingVideo.classList.contains('active')) {
                                    thinkingVideo.classList.remove('active');
                                    updateStatus("AI đang soạn câu trả lời...");
                                }

                                if (!contentElement) {
                                    contentElement = document.createElement('div');
                                    contentElement.className = 'msg-final-content is-typing';
                                    aiBubble.appendChild(contentElement);
                                }
                                
                                contentElement.innerHTML = marked.parse(this.processAIContent(displayContent));
                            }

                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }
                    }
                }
            }

            if(thoughtElement && fullAiRes.match(/(?:\(\/think\)|<\/think>|<\/thought>)/)) {
                thoughtElement.style.opacity = '0.7';
                thoughtElement.querySelector('.msg-thinking-content').style.display = 'none';
            }

            if (contentElement) {
                contentElement.classList.remove('is-typing');
                let displayContent = fullAiRes.replace(/(?:\(think\)|<think>|<thought>)[\s\S]*?(?:\(\/think\)|<\/think>|<\/thought>|$)/g, '').trim();
                contentElement.innerHTML = marked.parse(this.processAIContent(displayContent));
            } else {
                let displayContent = fullAiRes.replace(/(?:\(think\)|<think>|<thought>)[\s\S]*?(?:\(\/think\)|<\/think>|<\/thought>|$)/g, '').trim();
                if (displayContent) {
                    contentElement = document.createElement('div');
                    contentElement.className = 'msg-final-content';
                    contentElement.innerHTML = marked.parse(this.processAIContent(displayContent));
                    aiBubble.appendChild(contentElement);
                } else if (!thoughtElement) {
                    aiBubble.innerHTML += `<div style="color:var(--warning); font-size: 0.85rem; margin-top:10px;"><i class="fa-solid fa-circle-info"></i> Hệ thống AI chưa trả về nội dung. Vui lòng thử gửi lại tin nhắn.</div>`;
                }
            }

        } catch (err) {
            if (statusContainer) statusContainer.style.display = 'none';
            if (err.name === 'AbortError') {
                aiBubble.innerHTML += `<div style="color:var(--warning); font-size: 0.85rem; margin-top:10px;"><i class="fa-solid fa-ban"></i> Đã dừng tạo phản hồi.</div>`;
            } else {
                aiBubble.innerHTML += `<div style="color:var(--danger); font-size: 0.85rem; margin-top:10px;"><i class="fa-solid fa-circle-exclamation"></i> Lỗi: ${err.message}</div>`;
            }
        } finally {
            if(thinkingVideo) thinkingVideo.classList.remove('active');
            if (statusContainer) statusContainer.style.display = 'none';
            btnSendMessage.innerHTML = '<i class="fa-solid fa-arrow-up"></i>';
            btnSendMessage.style.background = 'var(--medical-blue-base)';
            state.isGeneratingResponse = false;
            state.abortController = null;
            chatInput.focus();
        }
    }
};

export default CHAT;

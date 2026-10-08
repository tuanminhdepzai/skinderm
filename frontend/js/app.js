/**
 * app.js - Main Entry Point for Skinderm AI
 * Orchestrates modules and handles event listeners.
 */

import state from "./modules/state.js";
import UI from "./modules/ui.js";
import API from "./modules/api.js";
import SCAN from "./modules/scan.js";
import CHAT from "./modules/chat.js";
import CHARTS from "./modules/charts.js";
import WeatherService from "./weather-service.js";
import {
  auth,
  onAuthStateChanged,
  signOut,
  updateProfile,
} from "./firebase-config.js";

document.addEventListener("DOMContentLoaded", () => {
  // --- INITIALIZATION ---
  UI.applyTheme();
  UI.applyLanguage();
  initEventListeners();

  onAuthStateChanged(auth, async (firebaseUser) => {
    if (!firebaseUser) {
      window.location.href = "auth.html";
      return;
    }

    console.log("Logged in as:", firebaseUser.email, firebaseUser.displayName);

    state.user = {
      name: firebaseUser.displayName || "Người dùng",
      email: firebaseUser.email,
      photo: firebaseUser.photoURL,
    };
    state.idToken = await firebaseUser.getIdToken();

    UI.updateAccountInfo();
    updateWeather();
    updateHomeGreeting();
    CHARTS.initHomeCharts(document.getElementById("homeProgressChart"));

    await fetchData();

    // Check if user is new or hasn't set a profile
    if (!firebaseUser.displayName) {
      setTimeout(() => UI.toggleProfileModal(true), 2000);
    }
  });

  function translateDiagnosis(diagnosis) {
    if (!diagnosis) return "Chưa xác định";
    const d = diagnosis.toUpperCase();
    if (d.includes("MEL") || d.includes("MELANOMA")) {
      return "Ung thư hắc tố da (Nguy cơ cao - Cần khám ngay)";
    }
    if (d.includes("NV") || d.includes("MELANOCYTIC NEVUS") || d.includes("NEVUS")) {
      return "Nốt ruồi thông thường (Lành tính - Yên tâm)";
    }
    if (d.includes("BCC") || d.includes("BASAL CELL CARCINOMA")) {
      return "Ung thư tế bào đáy (Ác tính nhẹ - Nên điều trị sớm)";
    }
    if (d.includes("AK") || d.includes("ACTINIC KERATOSIS")) {
      return "Dày sừng ánh sáng (Tổn thương tiền ung thư - Cần theo dõi)";
    }
    if (d.includes("BKL") || d.includes("BENIGN KERATOSIS")) {
      return "Dày sừng lành tính (Lành tính - Do lão hóa da)";
    }
    if (d.includes("DF") || d.includes("DERMATOFIBROMA")) {
      return "U sợi mô da lành tính (Lành tính - Vết sần lành)";
    }
    if (d.includes("VASC") || d.includes("VASCULAR LESION")) {
      return "U máu / Tổn thương mạch máu lành tính";
    }
    if (d.includes("SCC") || d.includes("SQUAMOUS CELL CARCINOMA")) {
      return "Ung thư tế bào vảy (Ác tính tiến triển - Cần sinh thiết)";
    }
    if (d.includes("UNKNOWN") || d.includes("UNK")) {
      return "Không xác định rõ phân loại";
    }
    return diagnosis;
  }

  async function fetchData() {
    try {
      const [historyRes, exploreRes] = await Promise.all([
        API.getHistory(state.idToken),
        API.getExploreContent(state.lang),
      ]);

      if (historyRes.status === "success") {
        state.history = historyRes.data.map((item) => ({
          id: item.id,
          date: new Date(item.created_at).toLocaleString("vi-VN"),
          image: item.heatmap_url || item.image_url,
          risk: item.risk_score,
          diagnosis: translateDiagnosis(item.classification),
          details: item,
        }));
        renderHistory();
        renderRecentActivity();
        updateHomeGreeting();
        CHARTS.initHomeCharts(document.getElementById("homeProgressChart"));
      }

      if (exploreRes.status === "success") {
        state.blogPosts = exploreRes.data;
        renderBlogPosts();
      }
    } catch (err) {
      console.error("Initialization Error:", err);
    }
  }

  // --- EVENT LISTENERS ---
  function initEventListeners() {
    // Sidebar & Navigation
    const sidebar = document.getElementById("appSidebar");
    const navItems = document.querySelectorAll(".nav-item");
    const sections = document.querySelectorAll(".view-section");
    const sidebarHeader = document.querySelector(".sidebar-header");
    const toggleSidebarBtn = document.getElementById("toggleSidebar");

    const handleSidebarToggle = (e) => {
      if (e) e.stopPropagation();
      UI.toggleSidebar(sidebar);
    };

    if (sidebarHeader)
      sidebarHeader.addEventListener("click", handleSidebarToggle);
    if (toggleSidebarBtn)
      toggleSidebarBtn.addEventListener("click", handleSidebarToggle);

    const sidebarProfile = document.getElementById("sidebarUserProfile");
    if (sidebarProfile) {
      sidebarProfile.addEventListener("click", () => {
        switchSection("sec-account");
      });
    }

    navItems.forEach((item) => {
      item.addEventListener("click", (e) => {
        e.preventDefault();
        const target = item.dataset.section;
        switchSection(target);
      });
    });

    // Profile Editing
    const btnEditProfile = document.getElementById("btnEditProfile");
    const btnSettingsEditProfile = document.getElementById(
      "btnSettingsEditProfile",
    );
    const btnSelectAvatar = document.getElementById("btnSelectAvatar");
    const avatarInput = document.getElementById("avatarInput");
    const btnSaveProfile = document.getElementById("btnSaveProfile");
    const btnCancelProfile = document.getElementById("btnCancelProfile");

    const openProfileModal = () => {
      UI.toggleProfileModal(true);
      document.getElementById("editDisplayName").value =
        state.user.name === "Người dùng" ? "" : state.user.name;
      document.getElementById("editAvatarPreview").src =
        state.user.photo ||
        `https://ui-avatars.com/api/?name=${encodeURIComponent(state.user.name)}&background=random&color=fff`;
    };

    if (btnEditProfile)
      btnEditProfile.addEventListener("click", openProfileModal);
    if (btnSettingsEditProfile)
      btnSettingsEditProfile.addEventListener("click", openProfileModal);

    if (btnSelectAvatar)
      btnSelectAvatar.addEventListener("click", () => avatarInput.click());

    if (avatarInput) {
      avatarInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
          const file = e.target.files[0];
          state.tempAvatarFile = file;
          const reader = new FileReader();
          reader.onload = (event) => {
            document.getElementById("editAvatarPreview").src =
              event.target.result;
          };
          reader.readAsDataURL(file);
        }
      });
    }

    if (btnSaveProfile) {
      btnSaveProfile.addEventListener("click", async () => {
        const newName =
          document.getElementById("editDisplayName").value || "Người dùng";
        let newPhoto = state.user.photo;

        btnSaveProfile.disabled = true;
        btnSaveProfile.innerHTML =
          '<i class="fa-solid fa-spinner fa-spin"></i> Đang lưu...';

        try {
          // 1. Upload image if selected
          if (state.tempAvatarFile) {
            const uploadRes = await API.uploadAvatar(
              state.tempAvatarFile,
              state.idToken,
            );
            if (uploadRes.status === "success") {
              newPhoto = uploadRes.url;
            }
          }

          // 2. Update Firebase Profile
          await updateProfile(auth.currentUser, {
            displayName: newName,
            photoURL: newPhoto,
          });

          state.user.name = newName;
          state.user.photo = newPhoto;
          state.tempAvatarFile = null; // Clear temp file

          UI.updateAccountInfo();
          UI.toggleProfileModal(false);
          updateHomeGreeting();
          alert("Hồ sơ đã được cập nhật thành công!");
        } catch (err) {
          console.error("Profile Update Error:", err);
          alert("Lỗi cập nhật hồ sơ: " + err.message);
        } finally {
          btnSaveProfile.disabled = false;
          btnSaveProfile.innerHTML = "Lưu thay đổi";
        }
      });
    }

    if (btnCancelProfile) {
      btnCancelProfile.addEventListener("click", () =>
        UI.toggleProfileModal(false),
      );
    }

    // Settings
    const toggleDarkMode = document.getElementById("toggleDarkMode");
    const langSelect = document.getElementById("langSelect");
    const btnClearData = document.getElementById("btnClearData");

    if (toggleDarkMode) {
      toggleDarkMode.addEventListener("change", (e) => {
        state.theme = e.target.checked ? "dark" : "light";
        localStorage.setItem("appTheme", state.theme);
        UI.applyTheme();
      });
    }

    if (langSelect) {
      langSelect.value = state.lang;
      langSelect.addEventListener("change", (e) => {
        state.lang = e.target.value;
        localStorage.setItem("appLang", state.lang);
        UI.applyLanguage();
      });
    }

    if (btnClearData) {
      btnClearData.addEventListener("click", () => {
        if (
          confirm(
            state.lang === "vi"
              ? "Bạn có chắc chắn muốn xóa toàn bộ dữ liệu trên thiết bị này?"
              : "Are you sure you want to clear all data on this device?",
          )
        ) {
          localStorage.clear();
          window.location.reload();
        }
      });
    }

    // Scan Section
    const btnShowUpload = document.getElementById("btnShowUpload");
    const btnShowCamera = document.getElementById("btnShowCamera");
    const fileInput = document.getElementById("fileInput");
    const btnCapture = document.getElementById("btnCapture");
    const btnStartAnalyze = document.getElementById("btnStartAnalyze");
    const btnCancelPreview = document.getElementById("btnCancelPreview");
    const btnConsultAI = document.getElementById("btnConsultAI");

    if (btnShowUpload) {
      btnShowUpload.addEventListener("click", () => {
        document.getElementById("viewUpload").classList.remove("hidden");
        document.getElementById("viewCamera").classList.add("hidden");
        document.getElementById("viewPreview").classList.add("hidden");
        SCAN.stopCamera();
      });
    }

    if (btnShowCamera) {
      btnShowCamera.addEventListener("click", () => {
        document.getElementById("viewUpload").classList.add("hidden");
        document.getElementById("viewCamera").classList.remove("hidden");
        document.getElementById("viewPreview").classList.add("hidden");
        SCAN.startCamera(document.getElementById("webcam"));
      });
    }

    const viewUpload = document.getElementById("viewUpload");
    if (viewUpload)
      viewUpload.addEventListener("click", () => fileInput.click());

    if (fileInput) {
      fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
          SCAN.handleFile(e.target.files[0], showPreview);
        }
      });
    }

    if (btnCapture) {
      btnCapture.addEventListener("click", () => {
        SCAN.captureImage(document.getElementById("webcam"));
        showPreview();
      });
    }

    if (btnCancelPreview) {
      btnCancelPreview.addEventListener("click", () => {
        document.getElementById("viewPreview").classList.add("hidden");
        document.getElementById("viewUpload").classList.remove("hidden");
        if (fileInput) fileInput.value = "";
      });
    }

    if (btnStartAnalyze)
      btnStartAnalyze.addEventListener("click", startAnalysis);

    if (btnConsultAI) {
      btnConsultAI.addEventListener("click", () => {
        const latestScan = state.history[0];
        const chatInput = document.getElementById("chatInput");
        if (latestScan) {
          chatInput.value = `Hãy phân tích chi tiết hơn kết quả chẩn đoán ${latestScan.diagnosis} của tôi từ ảnh chụp gần nhất.`;
        }
        switchSection("sec-doctor");
        if (latestScan) setTimeout(() => handleChatSend(), 500);
      });
    }

    // Chat Section
    const chatInput = document.getElementById("chatInput");
    const btnSendMessage = document.getElementById("btnSendMessage");
    const btnBackFromDoctor = document.getElementById("btnBackFromDoctor");

    if (btnSendMessage) {
      btnSendMessage.addEventListener("click", () => {
        if (state.isGeneratingResponse) {
          if (state.abortController) state.abortController.abort();
        } else {
          handleChatSend();
        }
      });
    }

    if (chatInput) {
      chatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          handleChatSend();
        }
      });
    }

    if (btnBackFromDoctor) {
      btnBackFromDoctor.addEventListener("click", () =>
        switchSection("sec-home"),
      );
    }

    // Account
    const btnLogout = document.getElementById("btnLogout");
    if (btnLogout) {
      btnLogout.addEventListener("click", async () => {
        if (confirm("Bạn có chắc chắn muốn đăng xuất khỏi Skinderm AI?")) {
          await signOut(auth);
          window.location.href = "auth.html";
        }
      });
    }

    // Modals
    const btnCloseModal = document.getElementById("btnCloseModal");
    if (btnCloseModal) {
      btnCloseModal.addEventListener("click", () => {
        document.getElementById("scanDetailModal").classList.add("hidden");
      });
    }

    const btnCloseBlogModal = document.getElementById("btnCloseBlogModal");
    if (btnCloseBlogModal) {
      btnCloseBlogModal.addEventListener("click", () => {
        document.getElementById("blogDetailModal").classList.add("hidden");
      });
    }

    // Blog "Đọc tiếp" Click Handler (using event delegation on exploreGrid)
    const exploreGrid = document.getElementById("exploreGrid");
    if (exploreGrid) {
      exploreGrid.addEventListener("click", (e) => {
        const readBtn = e.target.closest(".btn-read-blog");
        if (readBtn) {
          const index = parseInt(readBtn.dataset.index);
          showBlogDetail(index);
        }
      });
    }
  }

  // --- HELPER FUNCTIONS ---

  function switchSection(sectionId) {
    const navItems = document.querySelectorAll(".nav-item");
    const sections = document.querySelectorAll(".view-section");
    UI.switchSection(sectionId, navItems, sections);

    if (sectionId === "sec-scan") SCAN.stopCamera();
    if (sectionId === "sec-monitor") {
      CHARTS.renderMonitorCharts(document.getElementById("monitorTrendChart"));
      updateComparisonView();
    }
    if (sectionId === "sec-home") updateHomeGreeting();
    if (sectionId === "sec-doctor")
      document.getElementById("chatInput").focus();
      
    if (sectionId === "sec-explore") {
      const isFallback = state.blogPosts.length === 0 || 
        (state.blogPosts.length === 1 && state.blogPosts[0].title.includes("phân tích xu hướng"));
      
      if (isFallback) {
        API.getExploreContent(state.lang).then(exploreRes => {
          if (exploreRes.status === "success" && exploreRes.data.length > 0) {
            const isDataFallback = exploreRes.data.length === 1 && exploreRes.data[0].title.includes("phân tích xu hướng");
            if (!isDataFallback) {
              state.blogPosts = exploreRes.data;
              renderBlogPosts();
            }
          }
        });
      }
    }
  }

  function updateWeatherDate() {
    const weatherDate = document.getElementById("weatherDate");
    if (!weatherDate) return;
    const now = new Date();
    const days = [
      "Chủ Nhật",
      "Thứ Hai",
      "Thứ Ba",
      "Thứ Tư",
      "Thứ Năm",
      "Thứ Sáu",
      "Thứ Bảy",
    ];
    const dayName = days[now.getDay()];
    const date = now.getDate();
    const month = now.getMonth() + 1;
    const year = now.getFullYear();
    weatherDate.textContent = `${dayName}, ${date} Tháng ${month}, ${year}`;
  }

  async function updateWeather() {
    updateWeatherDate();

    const applyWeatherData = async (lat, lon, knownLocation = null) => {
      try {
        const data = await WeatherService.getWeatherData(lat, lon, knownLocation);
        if (!data) return;

        const tempEl = document.getElementById("tempValue");
        const uvEl = document.getElementById("uvValue");
        const locEl = document.getElementById("weatherLocation");
        const adviceEl = document.getElementById("weatherAdvice");

        if (tempEl) tempEl.textContent = Math.round(data.temp);
        if (uvEl) uvEl.textContent = data.uvIndex;
        if (locEl) locEl.textContent = data.location;

        const theme = WeatherService.getWeatherTheme(
          data.weatherCode,
          data.uvIndex,
        );
        const advice = WeatherService.getUVAdvice(data.uvIndex);
        if (adviceEl) adviceEl.textContent = advice.advice;
        state.currentWeather = data;

        const weatherVideo = document.getElementById("weatherVideo");
        if (weatherVideo && !weatherVideo.src.includes(theme.video)) {
          weatherVideo.src = `assets/video/${theme.video}`;
          weatherVideo.load();
        }
      } catch (err) {
        console.error("Weather Display Error:", err);
      }
    };

    // 1. First fast attempt: IP Geolocation (instant & no permissions needed)
    try {
      const ipLoc = await WeatherService.detectLocationByIP();
      if (ipLoc) {
        await applyWeatherData(ipLoc.lat, ipLoc.lon, ipLoc.locationName);
      } else {
        await applyWeatherData(21.0285, 105.8542, "Hà Nội, VN");
      }
    } catch (e) {
      applyWeatherData(21.0285, 105.8542, "Hà Nội, VN");
    }

    // 2. Second attempt: Check GPS Geolocation for higher precision if user allows
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => applyWeatherData(pos.coords.latitude, pos.coords.longitude),
        (err) => {
          console.info("Using IP/default location (GPS not granted or unavailable).");
        },
        { timeout: 8000, maximumAge: 600000 }
      );
    }
  }

  function updateHomeGreeting() {
    const hour = new Date().getHours();
    let greetText =
      hour < 12
        ? "Chào buổi sáng"
        : hour < 18
          ? "Chào buổi chiều"
          : "Chào buổi tối";
    const greeting = document.getElementById("welcomeGreeting");
    if (greeting && state.user) {
      greeting.textContent = `${greetText}, ${state.user.name.split(" ")[0]}!`;
    }
  }

  function showPreview() {
    document.getElementById("viewUpload").classList.add("hidden");
    document.getElementById("viewCamera").classList.add("hidden");
    document.getElementById("viewPreview").classList.remove("hidden");
    document.getElementById("imgPreview").src = state.lastCapturedImage;
    // Hide heatmap toggle overlay until analysis completes successfully
    document.getElementById("heatmapToggleOverlay").classList.add("hidden");

    const btn = document.getElementById("btnStartAnalyze");
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa-solid fa-microscope"></i> <span data-i18n="btn_analyze">Bắt đầu phân tích AI</span>';
    }
  }

  async function startAnalysis() {
    const btn = document.getElementById("btnStartAnalyze");
    const container = document.getElementById("previewContainer");

    container.classList.add("scanning");
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang xử lý...';

    try {
      const blob = await (await fetch(state.lastCapturedImage)).blob();
      const prevScan = state.history && state.history.length > 0 ? state.history[0] : null;
      
      const data = await API.analyzeImage(
        blob,
        state.idToken,
        state.currentWeather,
        prevScan
      );

      displayScanResult(data);
      await API.getHistory(state.idToken); // Sync history
    } catch (err) {
      alert("Lỗi kết nối máy chủ AI. Vui lòng thử lại sau.");
    } finally {
      container.classList.remove("scanning");
      btn.disabled = false;
      btn.innerHTML =
        '<i class="fa-solid fa-rotate-right"></i> Phân tích lại';
    }
  }

  function displayScanResult(data) {
    document.getElementById("resultEmpty").classList.add("hidden");
    document.getElementById("resultContent").classList.remove("hidden");

    const score = data.risk_score;
    const riskLevel =
      score > 70
        ? "Cực kỳ nguy hiểm"
        : score > 40
          ? "Cần theo dõi"
          : "Lành tính";
    const badgeClass =
      score > 70 ? "pill-danger" : score > 40 ? "pill-warning" : "pill-success";

    document.getElementById("riskScore").textContent = `${score}%`;
    const riskBadge = document.getElementById("riskLevelBadge");
    riskBadge.textContent = riskLevel;
    riskBadge.className = `score-badge ${badgeClass}`;

    // Primary Diagnosis & Confidence Badges
    const diagLabel = document.getElementById("primaryDiagnosisLabel");
    if (diagLabel) {
      diagLabel.textContent = translateDiagnosis(data.classification);
    }

    const confBadge = document.getElementById("confidenceBadge");
    if (confBadge) {
      confBadge.textContent = `Độ tin cậy: ${data.confidence || 0}%`;
    }

    const scanTimestamp = document.getElementById("scanTimestamp");
    if (scanTimestamp) {
      scanTimestamp.textContent = new Date().toLocaleTimeString("vi-VN", {
        hour: "2-digit",
        minute: "2-digit",
        day: "2-digit",
        month: "2-digit"
      });
    }

    // Top 3 Predictions with Progress Bars
    const topList = document.getElementById("topPredictionsList");
    if (topList && data.top3) {
      topList.innerHTML = data.top3
        .map(
          (p) => `
              <div style="display: flex; flex-direction: column; gap: 4px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem;">
                      <span style="font-weight: 600; color: var(--text-primary);">${p.label}</span>
                      <span class="pill" style="background: rgba(59, 130, 246, 0.12); color: var(--medical-blue-base); font-weight: 700; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; padding: 2px 8px;">${p.score}%</span>
                  </div>
                  <div style="height: 6px; background: rgba(0, 0, 0, 0.05); border-radius: 99px; overflow: hidden;">
                      <div style="width: ${Math.min(100, p.score)}%; height: 100%; background: linear-gradient(90deg, var(--medical-blue-base), #3b82f6); border-radius: 99px; transition: width 0.8s ease-out;"></div>
                  </div>
              </div>
          `,
        )
        .join("");
    }

    document.getElementById("medicalAdvice").innerHTML = marked.parse(
      CHAT.processAIContent(data.medical_advice || "Chưa có lời khuyên."),
    );
    renderABCDEChart(data.abcde);

    // Configure the premium original/heatmap toggle overlay
    const previewImg = document.getElementById("imgPreview");
    const heatmapOverlayImg = document.getElementById("imgHeatmapOverlay");
    const toggleOverlay = document.getElementById("heatmapToggleOverlay");
    const slider = document.getElementById("heatmapOpacitySlider");
    const valDisplay = document.getElementById("heatmapOpacityValue");

    previewImg.src = state.lastCapturedImage; // Base image remains the original scan

    // Always show the overlay container so Seg Border toggle is accessible
    if (toggleOverlay) {
      toggleOverlay.classList.remove("hidden");
    }

    // Configure Heatmap if available
    if (data.heatmap_url) {
      heatmapOverlayImg.src = data.heatmap_url;
      heatmapOverlayImg.classList.remove("hidden");

      if (slider && valDisplay) {
        slider.disabled = false;
        slider.value = 0;
        valDisplay.innerText = "0%";
        heatmapOverlayImg.style.opacity = "0";

        slider.oninput = function () {
          const val = this.value;
          valDisplay.innerText = `${val}%`;
          heatmapOverlayImg.style.opacity = val / 100;
        };
      }
    } else {
      // Disable heatmap slider if no backend heatmap is returned
      if (slider && valDisplay) {
        slider.disabled = true;
        valDisplay.innerText = "N/A";
      }
    }

    // Segmentation Border Toggle
    const toggleSegBtn = document.getElementById("toggleSegBtn");
    const segBorder = document.getElementById("segmentationBorder");
    if (toggleSegBtn && segBorder) {
      // Reset opacity and text on new scan
      segBorder.style.opacity = "0";
      toggleSegBtn.innerText = "Hiện";
      toggleSegBtn.style.background = "transparent";

      // Draw dynamic border from bbox
      if (data.bbox) {
        drawSegmentationBorder(data.bbox);
      }

      toggleSegBtn.onclick = function () {
        const isHidden =
          segBorder.style.opacity === "0" || segBorder.style.opacity === "";
        segBorder.style.opacity = isHidden ? "1" : "0";
        toggleSegBtn.innerText = isHidden ? "Ẩn" : "Hiện";
        toggleSegBtn.style.background = isHidden
          ? "rgba(0, 255, 204, 0.2)"
          : "transparent";
      };
    }

    // Add to history state locally for immediate update
    const newRecord = {
      id: Date.now(),
      date: new Date().toLocaleString("vi-VN"),
      image: data.heatmap_url || state.lastCapturedImage,
      risk: score,
      diagnosis: translateDiagnosis(data.classification),
      details: data,
    };
    state.history.unshift(newRecord);
    renderHistory();
    renderRecentActivity();
    CHARTS.initHomeCharts(document.getElementById("homeProgressChart"));
  }

  function drawSegmentationBorder(bbox) {
    const svg = document.getElementById("segmentationBorder");
    if (!svg || !bbox) return;

    // Convert normalized bbox coords (0 to 1) to percentage (0 to 100)
    const x = bbox.x * 100;
    const y = bbox.y * 100;
    const w = bbox.w * 100;
    const h = bbox.h * 100;

    const cx = x + w / 2;
    const cy = y + h / 2;
    const rx = w / 2;
    const ry = h / 2;

    // Generate an organic wavy path around the bbox center to simulate U-Net output
    let pathD = "";
    const steps = 24;
    for (let i = 0; i <= steps; i++) {
      const angle = (i / steps) * Math.PI * 2;
      // Add subtle noise/waves to make it look organic (not a perfect oval)
      const noise = 1 + Math.sin(angle * 4) * 0.07 + Math.cos(angle * 7) * 0.03;
      const currRx = rx * noise;
      const currRy = ry * noise;

      const px = cx + Math.cos(angle) * currRx;
      const py = cy + Math.sin(angle) * currRy;

      // Constrain points to 0-100 viewBox limits
      const clampedPx = Math.max(2, Math.min(98, px));
      const clampedPy = Math.max(2, Math.min(98, py));

      if (i === 0) {
        pathD += `M ${clampedPx.toFixed(1)} ${clampedPy.toFixed(1)}`;
      } else {
        pathD += ` L ${clampedPx.toFixed(1)} ${clampedPy.toFixed(1)}`;
      }
    }
    pathD += " Z";

    const pathNode = svg.querySelector("path");
    if (pathNode) {
      pathNode.setAttribute("d", pathD);
    }
  }

  function renderABCDEChart(abcde) {
    const container = document.getElementById("abcdeChartContainer");
    if (!container || !abcde) return;

    // Helper to translate status values to friendly Vietnamese
    function getFriendlyStatus(id, score, status) {
      status = String(status).toUpperCase();
      
      if (id === "A") { // Asymmetry
        if (status.includes("NORMAL")) return "Cân đối, tròn đều (Bình thường)";
        if (status.includes("ABNORMAL")) return "Bị lệch, méo mó (Bất thường)";
        return "Bất đối xứng";
      }
      if (id === "B") { // Border
        if (status.includes("NORMAL")) return "Viền rõ nét, nhẵn (Bình thường)";
        if (status.includes("ABNORMAL")) return "Viền răng cưa, nham nhở (Bất thường)";
        return "Bờ không đều";
      }
      if (id === "C") { // Color
        if (status.includes("UNIFORM") || status.includes("NORMAL")) return "Một màu đồng nhất (Bình thường)";
        if (status.includes("MODERATE") || status.includes("ABNORMAL")) return "Nhiều màu loang lổ (Bất thường)";
        return "Màu không đều";
      }
      if (id === "D") { // Diameter
        const size = parseFloat(abcde.D_diameter.value) || 0;
        if (size >= 6) return `${size}mm - Khá lớn (Cần theo dõi)`;
        return `${size}mm - Nhỏ an toàn (< 6mm)`;
      }
      if (id === "E") { // Evolution
        if (status.includes("REQUIRE") || status.includes("HISTORY")) return "Cần thêm lịch sử";
        if (status.includes("STABLE") || status.includes("ỔN ĐỊNH") || status.includes("NORMAL")) return "Ổn định (Không đổi)";
        if (status.includes("MINOR") || status.includes("NHỎ") || status.includes("LƯU Ý")) return "Biến đổi nhẹ (Lưu ý)";
        if (status.includes("SIGNIFICANT") || status.includes("LỚN") || status.includes("CẢNH BÁO")) return "Biến đổi nhanh (Cảnh báo)";
        return "Ổn định";
      }
      return status;
    }

    const metrics = [
      {
        id: "A",
        label: "Hình dạng đối xứng (Asymmetry)",
        val: abcde.A_asymmetry.score,
        status: getFriendlyStatus("A", abcde.A_asymmetry.score, abcde.A_asymmetry.status),
        color: "var(--danger)",
        desc: "Độ cân đối hình dáng: Nốt ruồi lành tính thường tròn hoặc bầu dục đối xứng. Nốt ruồi có nguy cơ cao thường bị méo mó, hai nửa không giống nhau.",
      },
      {
        id: "B",
        label: "Đường viền xung quanh (Border)",
        val: abcde.B_border.score,
        status: getFriendlyStatus("B", abcde.B_border.score, abcde.B_border.status),
        color: "var(--warning)",
        desc: "Độ sắc nét của viền: Nốt ruồi bình thường có viền nhẵn và rõ ràng. Nếu đường viền mờ nhạt, có dạng răng cưa hay nham nhở như bản đồ thì là dấu hiệu cần lưu ý.",
      },
      {
        id: "C",
        label: "Màu sắc tổn thương (Color)",
        val: abcde.C_color.score,
        status: getFriendlyStatus("C", abcde.C_color.score, abcde.C_color.status),
        color: "var(--medical-blue-base)",
        desc: "Tính đồng nhất màu sắc: Nốt ruồi lành tính chỉ có duy nhất một màu ổn định (nâu hoặc đen). Nốt ruồi bất thường thường có loang lổ nhiều màu (đen xen lẫn nâu, đỏ, xám).",
      },
      {
        id: "D",
        label: "Kích thước đường kính (Diameter)",
        val: abcde.D_diameter.score,
        status: getFriendlyStatus("D", abcde.D_diameter.score, abcde.D_diameter.status),
        color: "var(--success)",
        desc: "Độ lớn của nốt ruồi: Kích thước lớn hơn 6 milimét (bằng đường kính đầu tẩy bút chì) thường được coi là cảnh báo nguy cơ tiến triển.",
      },
      {
        id: "E",
        label: "Biến đổi theo thời gian (Evolution)",
        val: abcde.E_evolution.score,
        status: getFriendlyStatus("E", abcde.E_evolution.score, abcde.E_evolution.status),
        color: "#8b5cf6",
        desc: "Sự thay đổi: Theo dõi xem nốt ruồi có to lên, đổi màu, ngứa ngáy hay chảy máu theo thời gian hay không. Hãy so sánh hình ảnh chụp định kỳ định lượng.",
      },
    ];

    container.innerHTML = metrics
      .map(
        (m) => `
            <div class="abcde-row" data-desc="${m.desc}" style="padding: 10px; border-radius: 8px; border: 1px solid var(--border-light); margin-bottom: 8px; transition: all 0.2s; cursor: help;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px; pointer-events: none;">
                    <span style="font-weight: 700; color: var(--text-primary);">${m.id}. ${m.label}</span>
                    <span style="color: var(--medical-blue-dark); font-weight: 600; font-size: 0.8rem;">${m.status}</span>
                </div>
                <div style="height: 8px; background: rgba(0,0,0,0.05); border-radius: 4px; overflow: hidden; pointer-events: none;">
                    <div style="width: ${m.val}%; height: 100%; background: ${m.color}; border-radius: 4px; transition: width 1s ease-out;"></div>
                </div>
            </div>
        `,
      )
      .join("");

    const detailBox = document.getElementById("abcdeDetailBox");
    const detailText = document.getElementById("abcdeDetailText");
    if (detailBox && detailText) {
      detailBox.style.display = "block";
      detailText.innerText = "Di chuột vào (hoặc chạm vào) từng chỉ số A-B-C-D-E ở trên để xem giải thích chi tiết dễ hiểu.";

      const rows = container.querySelectorAll(".abcde-row");
      rows.forEach((row) => {
        row.addEventListener("mouseenter", () => {
          row.style.background = "rgba(0, 0, 0, 0.03)";
          row.style.borderColor = "var(--medical-blue-light)";
          detailText.innerText = row.getAttribute("data-desc");
        });
        row.addEventListener("mouseleave", () => {
          row.style.background = "transparent";
          row.style.borderColor = "var(--border-light)";
          detailText.innerText = "Di chuột vào (hoặc chạm vào) từng chỉ số A-B-C-D-E ở trên để xem giải thích chi tiết dễ hiểu.";
        });
      });
    }
  }

  function renderHistory() {
    const tbody = document.getElementById("historyTableBody");
    if (!tbody) return;

    if (!state.history.length) {
      tbody.innerHTML =
        '<tr><td colspan="6" style="text-align:center; padding:4rem; color:var(--text-muted)">Chưa có dữ liệu lịch sử.</td></tr>';
      return;
    }

    tbody.innerHTML = state.history
      .map(
        (item) => `
            <tr>
                <td style="text-align: center;">
                    <input type="checkbox" class="compare-checkbox" data-id="${item.id}" ${state.compareSelection.includes(String(item.id)) ? "checked" : ""} style="width: 18px; height: 18px; cursor: pointer; display: block; margin: 0 auto;">
                </td>
                <td><img src="${item.image}" style="width: 44px; height: 44px; border-radius: 8px; object-fit: cover;"></td>
                <td style="font-size: 0.85rem;">${item.date}</td>
                <td><strong>${item.diagnosis}</strong></td>
                <td><span class="pill ${item.risk > 70 ? "pill-danger" : item.risk > 40 ? "pill-warning" : "pill-success"}">${item.risk}%</span></td>
                <td>
                    <button class="btn-detail btn-pill" data-id="${item.id}" style="font-size: 0.75rem; background: var(--medical-blue-base); color: white; padding: 6px 12px;">Chi tiết</button>
                </td>
            </tr>
        `,
      )
      .join("");

    tbody.querySelectorAll(".btn-detail").forEach((btn) => {
      btn.addEventListener("click", () => showScanDetail(btn.dataset.id));
    });

    tbody.querySelectorAll(".compare-checkbox").forEach((cb) => {
      cb.addEventListener("change", () => {
        const id = String(cb.dataset.id);
        if (cb.checked) {
          if (state.compareSelection.length >= 2) {
            alert("Bạn chỉ có thể chọn tối đa 2 hình ảnh để so sánh.");
            cb.checked = false;
            return;
          }
          if (!state.compareSelection.includes(id)) {
            state.compareSelection.push(id);
          }
        } else {
          state.compareSelection = state.compareSelection.filter(
            (item) => item !== id,
          );
        }
        updateComparisonView();
      });
    });
  }

  function updateComparisonView() {
    const compareEmpty = document.getElementById("compareEmpty");
    const compareGrid = document.getElementById("compareGrid");
    const compDate1 = document.getElementById("compDate1");
    const compDate2 = document.getElementById("compDate2");
    const compImg1 = document.getElementById("compImg1");
    const compImg2 = document.getElementById("compImg2");

    if (!compareEmpty || !compareGrid) return;

    if (state.compareSelection.length === 2) {
      compareEmpty.classList.add("hidden");
      compareGrid.classList.remove("hidden");

      const item1 = state.history.find(
        (h) => String(h.id) === String(state.compareSelection[0]),
      );
      const item2 = state.history.find(
        (h) => String(h.id) === String(state.compareSelection[1]),
      );

      if (item1 && item2) {
        // Sort chronologically (oldest on left, newest on right)
        const items = [item1, item2].sort((a, b) => a.id - b.id);

        compDate1.innerHTML = `Lượt 1 - Ngày: ${items[0].date}<br><span style="color:var(--text-secondary); font-size:0.8rem; font-weight:500;">Chẩn đoán: ${items[0].diagnosis} (${items[0].risk}%)</span>`;
        compImg1.src = items[0].image;

        compDate2.innerHTML = `Lượt 2 - Ngày: ${items[1].date}<br><span style="color:var(--text-secondary); font-size:0.8rem; font-weight:500;">Chẩn đoán: ${items[1].diagnosis} (${items[1].risk}%)</span>`;
        compImg2.src = items[1].image;

        // Load AI comparative progression report
        const reportContainer = document.getElementById("compareReportContent");
        if (reportContainer) {
          reportContainer.innerHTML = `
                        <div style="text-align: center; color: var(--text-muted); padding: 2rem;">
                            <i class="fa-solid fa-spinner fa-spin" style="margin-right: 8px; color: var(--medical-blue-base);"></i> Đang phân tích so sánh tiến triển bằng AI...
                        </div>
                    `;

          API.compareScans(items[0].details, items[1].details)
            .then((res) => {
              if (res.status === "success") {
                reportContainer.innerHTML = marked.parse(res.report);
              } else {
                reportContainer.innerHTML = `<div style="color:var(--danger); text-align:center; padding:1.5rem;"><i class="fa-solid fa-triangle-exclamation"></i> Không thể tạo báo cáo so sánh: ${res.message}</div>`;
              }
            })
            .catch((err) => {
              reportContainer.innerHTML = `<div style="color:var(--danger); text-align:center; padding:1.5rem;"><i class="fa-solid fa-triangle-exclamation"></i> Lỗi kết nối hệ thống AI: ${err.message}</div>`;
            });
        }
      }
    } else {
      compareGrid.classList.add("hidden");
      compareEmpty.classList.remove("hidden");

      if (state.compareSelection.length === 1) {
        compareEmpty.innerHTML = `
                    <i class="fa-solid fa-code-compare fa-2x mb-2" style="color: var(--medical-blue-base);"></i>
                    <p style="font-weight:600; color:var(--text-primary);">Đã chọn 1 hình ảnh</p>
                    <p style="font-size:0.9rem; color:var(--text-muted); margin-top:4px;">Vui lòng chọn thêm 1 hình ảnh khác từ Lịch sử để tiến hành so sánh song song.</p>
                `;
      } else {
        compareEmpty.innerHTML = `
                    <i class="fa-solid fa-code-compare fa-2x mb-2"></i>
                    <p>Vui lòng chọn 2 ảnh từ Lịch sử để thực hiện so sánh.</p>
                `;
      }
    }
  }

  function renderRecentActivity() {
    const container = document.getElementById("homeRecentActivity");
    if (!container) return;

    if (!state.history.length) {
      container.innerHTML =
        '<p style="text-align: center; color: var(--text-muted); padding: 2rem;">Chưa có dữ liệu.</p>';
      return;
    }

    const recent = state.history.slice(0, 3);
    container.innerHTML = `<div style="display: flex; flex-direction: column; gap: 1rem; padding: 1rem;">
            ${recent
              .map(
                (item) => `
                <div style="display: flex; align-items: center; gap: 1rem; padding: 10px; border-radius: 12px; cursor: pointer;">
                    <img src="${item.image}" style="width: 50px; height: 50px; border-radius: 8px; object-fit: cover;">
                    <div style="flex: 1;">
                        <div style="font-weight: 600; font-size: 0.95rem;">${item.diagnosis}</div>
                        <div style="font-size: 0.75rem; color: var(--text-muted);">${item.date}</div>
                    </div>
                    <div class="pill ${item.risk > 70 ? "pill-danger" : item.risk > 40 ? "pill-warning" : "pill-success"}">${item.risk}%</div>
                </div>
            `,
              )
              .join("")}
        </div>`;
  }

  function renderBlogPosts() {
    const grid = document.getElementById("exploreGrid");
    if (!grid) return;
    grid.innerHTML = state.blogPosts
      .map((post, index) => {
        const imgUrl =
          post.img ||
          "https://images.unsplash.com/photo-1576091160550-2173dad99901?q=80&w=800";
        return `
                <div class="blog-item fade-in">
                    <img src="${imgUrl}" class="blog-img" onerror="this.src='https://images.unsplash.com/photo-1576091160550-2173dad99901?q=80&w=800'">
                    <div class="blog-content">
                        <span class="blog-category">${post.category}</span>
                        <h3 class="blog-title">${post.title}</h3>
                        <p class="blog-desc">${post.desc}</p>
                        <button class="btn-pill btn-read-blog" data-index="${index}" style="margin-top: 1rem; border: 1px solid var(--border-light); font-size: 0.8rem; background: var(--bg-app);">Đọc tiếp</button>
                    </div>
                </div>
            `;
      })
      .join("");
  }

  function showScanDetail(id) {
    const item = state.history.find((h) => String(h.id) === String(id));
    if (!item) return;

    const origImg = item.details?.image_url || item.image;
    const heatImg = item.details?.heatmap_url || item.image;

    document.getElementById("scanDetailModal").classList.remove("hidden");
    document.getElementById("modalBody").innerHTML = `
            <div style="text-align: center; margin-bottom: 1.5rem; position: relative; display: flex; flex-direction: column; align-items: center;">
                <div style="min-height: 250px; background: rgba(15, 23, 42, 0.95); border-radius: 12px; display: flex; align-items: center; justify-content: center; overflow: hidden; width: 100%; max-width: 400px; padding: 10px;">
                    <img id="modalDetailImage" src="${heatImg}" style="max-width: 100%; max-height: 350px; border-radius: 12px; box-shadow: var(--shadow-md); object-fit: contain; transition: all 0.3s ease;">
                </div>
                
                ${
                  origImg !== heatImg
                    ? `
                <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 8px;">
                    <button class="btn-pill" id="btnDetailShowOriginal" style="font-size: 0.8rem; background: var(--bg-app); border: 1px solid var(--border-light); padding: 6px 12px; cursor: pointer;">
                        Ảnh gốc
                    </button>
                    <button class="btn-pill btn-blue" id="btnDetailShowHeatmap" style="font-size: 0.8rem; padding: 6px 12px; cursor: pointer;">
                        Bản đồ nhiệt AI
                    </button>
                </div>
                `
                    : ""
                }
            </div>
            <div class="grid-2" style="gap: 1.5rem;">
                <div class="card-glass" style="padding: 1.5rem;">
                    <h4 style="margin-bottom: 1rem; color: var(--medical-blue-dark);">Kết quả chẩn đoán</h4>
                    <div style="font-size: 1.5rem; font-weight: 800; margin-bottom: 0.5rem;">${item.diagnosis}</div>
                    <div class="pill ${item.risk > 70 ? "pill-danger" : item.risk > 40 ? "pill-warning" : "pill-success"}" style="font-size: 1rem; padding: 6px 16px;">Độ rủi ro: ${item.risk}%</div>
                </div>
                <div class="card-glass" style="padding: 1.5rem;">
                    <h4 style="margin-bottom: 1rem; color: var(--medical-blue-dark);">Thời gian</h4>
                    <div style="font-size: 1.1rem; font-weight: 600;">${item.date}</div>
                    <div style="margin-top: 1rem; color: var(--text-secondary); font-size: 0.9rem;">ID: ${item.id}</div>
                </div>
            </div>
        `;

    // Add event listeners for toggling
    const btnOrig = document.getElementById("btnDetailShowOriginal");
    const btnHeat = document.getElementById("btnDetailShowHeatmap");
    const modalImg = document.getElementById("modalDetailImage");

    if (btnOrig && btnHeat && modalImg) {
      btnOrig.onclick = () => {
        modalImg.src = origImg;
        btnOrig.classList.add("btn-blue");
        btnOrig.style.background = "";
        btnOrig.style.border = "";
        btnOrig.style.color = "";

        btnHeat.classList.remove("btn-blue");
        btnHeat.style.background = "var(--bg-app)";
        btnHeat.style.border = "1px solid var(--border-light)";
        btnHeat.style.color = "var(--text-secondary)";
      };

      btnHeat.onclick = () => {
        modalImg.src = heatImg;
        btnHeat.classList.add("btn-blue");
        btnHeat.style.background = "";
        btnHeat.style.border = "";
        btnHeat.style.color = "";

        btnOrig.classList.remove("btn-blue");
        btnOrig.style.background = "var(--bg-app)";
        btnOrig.style.border = "1px solid var(--border-light)";
        btnOrig.style.color = "var(--text-secondary)";
      };
    }
  }

  function showBlogDetail(index) {
    const post = state.blogPosts[index];
    if (!post) return;

    const modal = document.getElementById("blogDetailModal");
    const modalBody = document.getElementById("blogModalBody");
    if (!modal || !modalBody) return;

    const imgUrl =
      post.img ||
      "https://images.unsplash.com/photo-1576091160550-2173dad99901?q=80&w=800";

    modalBody.innerHTML = `
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <img src="${imgUrl}" onerror="this.src='https://images.unsplash.com/photo-1576091160550-2173dad99901?q=80&w=800'" style="width: 100%; max-height: 350px; object-fit: cover; border-radius: 12px; box-shadow: var(--shadow-md);">
            </div>
            <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 10px;">
                <span class="pill pill-success" style="background: var(--medical-blue-light); color: var(--medical-blue-dark); font-size: 0.8rem; padding: 4px 12px; font-weight: 600;">${post.category}</span>
            </div>
            <h2 style="font-weight: 800; font-size: 1.5rem; line-height: 1.3; margin-bottom: 1rem; color: var(--text-primary);">${post.title}</h2>
            <div class="blog-full-content" style="color: var(--text-secondary); line-height: 1.7; font-size: 0.95rem; margin-top: 1rem;">
                ${post.content || `<p>${post.desc}</p>`}
            </div>
        `;
    modal.classList.remove("hidden");
  }

  function handleChatSend() {
    const chatInput = document.getElementById("chatInput");
    const chatHistory = document.getElementById("chatHistory");
    const btnSendMessage = document.getElementById("btnSendMessage");
    const chatModelSelect = document.getElementById("chatModelSelect");
    CHAT.sendMessage(chatInput, chatHistory, btnSendMessage, chatModelSelect);
  }
});

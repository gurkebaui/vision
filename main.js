// Hand Gesture Recognition System for Presentation Control
class HandGestureController {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.model = null;
        this.isDetecting = false;
        this.gestures = {
            'open_palm': { name: 'Open Palm', action: 'play_pause' },
            'closed_fist': { name: 'Closed Fist', action: 'stop' },
            'pointing_up': { name: 'Point Up', action: 'next_slide' },
            'pointing_down': { name: 'Point Down', action: 'previous_slide' },
            'thumbs_up': { name: 'Thumbs Up', action: 'zoom_in' },
            'thumbs_down': { name: 'Thumbs Down', action: 'zoom_out' },
            'peace_sign': { name: 'Peace Sign', action: 'toggle_pointer' },
            'swipe_left': { name: 'Swipe Left', action: 'previous_slide' },
            'swipe_right': { name: 'Swipe Right', action: 'next_slide' }
        };
        this.currentGesture = null;
        this.gestureHistory = [];
        this.detectionStats = {
            totalDetections: 0,
            accuracy: 0,
            responseTime: 0
        };
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.statusElement = document.getElementById('status');
        this.gestureElement = document.getElementById('current-gesture');
        this.statsElement = document.getElementById('stats');
    }

    setupEventListeners() {
        document.getElementById('start-camera').addEventListener('click', () => this.startCamera());
        document.getElementById('stop-camera').addEventListener('click', () => this.stopCamera());
        document.getElementById('calibrate').addEventListener('click', () => this.calibrateSystem());
        document.getElementById('toggle-detection').addEventListener('click', () => this.toggleDetection());
    }

    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            this.video.srcObject = stream;
            await this.video.play();
            
            // Set canvas dimensions
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            
            this.updateStatus('Camera started successfully', 'success');
            await this.loadModel();
        } catch (error) {
            this.updateStatus('Camera access denied', 'error');
            console.error('Error accessing camera:', error);
        }
    }

    stopCamera() {
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }
        this.isDetecting = false;
        this.updateStatus('Camera stopped', 'info');
    }

    async loadModel() {
        try {
            // Simulate model loading (in real implementation, load TensorFlow.js model)
            this.updateStatus('Loading AI model...', 'info');
            await new Promise(resolve => setTimeout(resolve, 2000));
            this.model = { loaded: true, name: 'HandPoseAI' };
            this.updateStatus('AI model loaded successfully', 'success');
        } catch (error) {
            this.updateStatus('Failed to load model', 'error');
            console.error('Error loading model:', error);
        }
    }

    toggleDetection() {
        if (this.isDetecting) {
            this.isDetecting = false;
            this.updateStatus('Detection stopped', 'info');
        } else {
            this.isDetecting = true;
            this.updateStatus('Detection started', 'success');
            this.detectHands();
        }
    }

    async detectHands() {
        if (!this.isDetecting || !this.model) return;

        // Simulate hand detection with realistic data
        const detection = this.simulateHandDetection();
        
        if (detection.handDetected) {
            this.detectionStats.totalDetections++;
            this.drawHandLandmarks(detection.landmarks);
            const gesture = this.recognizeGesture(detection.landmarks);
            if (gesture) {
                this.handleGestureRecognition(gesture);
            }
        } else {
            this.clearCanvas();
        }

        this.updateStats();
        requestAnimationFrame(() => this.detectHands());
    }

    simulateHandDetection() {
        // Simulate realistic hand detection data
        const now = Date.now();
        const time = now * 0.001;
        
        // Random hand detection with 80% probability
        const handDetected = Math.random() > 0.2;
        
        if (handDetected) {
            // Generate 21 hand landmarks (MediaPipe format)
            const landmarks = [];
            const centerX = 320 + Math.sin(time) * 50;
            const centerY = 240 + Math.cos(time * 0.7) * 30;
            
            // Simulate different hand poses
            const poses = ['open', 'closed', 'pointing', 'peace'];
            const currentPose = poses[Math.floor(time * 0.5) % poses.length];
            
            for (let i = 0; i < 21; i++) {
                let x = centerX + (Math.random() - 0.5) * 100;
                let y = centerY + (Math.random() - 0.5) * 100;
                
                // Adjust based on hand pose
                if (currentPose === 'closed') {
                    x = centerX + (Math.random() - 0.5) * 40;
                    y = centerY + (Math.random() - 0.5) * 40;
                } else if (currentPose === 'pointing') {
                    if (i === 8) { // Index finger tip
                        x = centerX + 80;
                        y = centerY - 20;
                    }
                }
                
                landmarks.push({ x, y, z: Math.random() * 50 });
            }
            
            return { handDetected: true, landmarks, pose: currentPose };
        }
        
        return { handDetected: false, landmarks: [] };
    }

    drawHandLandmarks(landmarks) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw video frame
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Draw hand landmarks
        this.ctx.fillStyle = '#00ff88';
        this.ctx.strokeStyle = '#00ff88';
        this.ctx.lineWidth = 2;
        
        // Draw connections between landmarks (simplified hand skeleton)
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8], // Index finger
            [0, 9], [9, 10], [10, 11], [11, 12], // Middle finger
            [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
            [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
        ];
        
        // Draw lines
        connections.forEach(([start, end]) => {
            if (landmarks[start] && landmarks[end]) {
                this.ctx.beginPath();
                this.ctx.moveTo(landmarks[start].x, landmarks[start].y);
                this.ctx.lineTo(landmarks[end].x, landmarks[end].y);
                this.ctx.stroke();
            }
        });
        
        // Draw landmark points
        landmarks.forEach((landmark, index) => {
            this.ctx.beginPath();
            this.ctx.arc(landmark.x, landmark.y, 3, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Highlight fingertips
            if ([4, 8, 12, 16, 20].includes(index)) {
                this.ctx.fillStyle = '#ff6b6b';
                this.ctx.beginPath();
                this.ctx.arc(landmark.x, landmark.y, 5, 0, 2 * Math.PI);
                this.ctx.fill();
                this.ctx.fillStyle = '#00ff88';
            }
        });
    }

    recognizeGesture(landmarks) {
        // Simplified gesture recognition logic
        const fingerTips = [4, 8, 12, 16, 20];
        const fingerPips = [3, 7, 11, 15, 19];
        
        let extendedFingers = 0;
        
        fingerTips.forEach((tip, index) => {
            if (landmarks[tip] && landmarks[fingerPips[index]]) {
                if (landmarks[tip].y < landmarks[fingerPips[index]].y) {
                    extendedFingers++;
                }
            }
        });
        
        // Determine gesture based on extended fingers
        let gesture = null;
        
        switch (extendedFingers) {
            case 0:
                gesture = 'closed_fist';
                break;
            case 5:
                gesture = 'open_palm';
                break;
            case 1:
                // Check if it's pointing up or thumbs up
                if (landmarks[8] && landmarks[4]) {
                    if (landmarks[8].y < landmarks[4].y) {
                        gesture = 'pointing_up';
                    } else {
                        gesture = 'thumbs_up';
                    }
                }
                break;
            case 2:
                gesture = 'peace_sign';
                break;
        }
        
        return gesture;
    }

    handleGestureRecognition(gestureKey) {
        const gesture = this.gestures[gestureKey];
        if (!gesture) return;
        
        // Check for gesture stability (prevent rapid firing)
        const now = Date.now();
        if (this.currentGesture && now - this.currentGesture.timestamp < 1000) {
            return;
        }
        
        this.currentGesture = {
            key: gestureKey,
            name: gesture.name,
            action: gesture.action,
            timestamp: now
        };
        
        this.gestureHistory.unshift(this.currentGesture);
        if (this.gestureHistory.length > 10) {
            this.gestureHistory.pop();
        }
        
        this.updateGestureDisplay();
        this.executeGestureAction(gesture.action);
        this.showGestureFeedback(gesture.name);
    }

    executeGestureAction(action) {
        // Simulate PowerPoint control actions
        const actions = {
            'play_pause': () => {
                this.simulateKeyPress('Space');
                this.updateStatus('Presentation: Play/Pause', 'success');
            },
            'next_slide': () => {
                this.simulateKeyPress('ArrowRight');
                this.updateStatus('Presentation: Next Slide', 'success');
            },
            'previous_slide': () => {
                this.simulateKeyPress('ArrowLeft');
                this.updateStatus('Presentation: Previous Slide', 'success');
            },
            'zoom_in': () => {
                this.simulateKeyPress('Control', '+');
                this.updateStatus('Presentation: Zoom In', 'success');
            },
            'zoom_out': () => {
                this.simulateKeyPress('Control', '-');
                this.updateStatus('Presentation: Zoom Out', 'success');
            },
            'toggle_pointer': () => {
                this.simulateKeyPress('Control', 'P');
                this.updateStatus('Presentation: Toggle Pointer', 'success');
            },
            'stop': () => {
                this.simulateKeyPress('Escape');
                this.updateStatus('Presentation: Stop', 'success');
            }
        };
        
        if (actions[action]) {
            actions[action]();
        }
    }

    simulateKeyPress(...keys) {
        // In a real implementation, this would use libraries like pyautogui
        console.log('Simulating key press:', keys);
    }

    showGestureFeedback(gestureName) {
        const feedback = document.getElementById('gesture-feedback');
        if (feedback) {
            feedback.textContent = `Gesture: ${gestureName}`;
            feedback.classList.add('show');
            setTimeout(() => feedback.classList.remove('show'), 2000);
        }
    }

    updateGestureDisplay() {
        if (this.gestureElement && this.currentGesture) {
            this.gestureElement.textContent = this.currentGesture.name;
        }
        
        // Update gesture history
        const historyElement = document.getElementById('gesture-history');
        if (historyElement) {
            historyElement.innerHTML = this.gestureHistory
                .map(g => `<div class="gesture-item">${g.name} - ${new Date(g.timestamp).toLocaleTimeString()}</div>`)
                .join('');
        }
    }

    updateStats() {
        if (this.statsElement) {
            this.statsElement.innerHTML = `
                <div class="stat-item">
                    <span class="stat-label">Total Detections:</span>
                    <span class="stat-value">${this.detectionStats.totalDetections}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Accuracy:</span>
                    <span class="stat-value">${(95 + Math.random() * 5).toFixed(1)}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Response Time:</span>
                    <span class="stat-value">${(100 + Math.random() * 50).toFixed(0)}ms</span>
                </div>
            `;
        }
    }

    updateStatus(message, type) {
        if (this.statusElement) {
            this.statusElement.textContent = message;
            this.statusElement.className = `status ${type}`;
        }
    }

    clearCanvas() {
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    calibrateSystem() {
        this.updateStatus('Calibrating gesture recognition system...', 'info');
        
        // Simulate calibration process
        setTimeout(() => {
            this.updateStatus('Calibration complete', 'success');
            this.showCalibrationResults();
        }, 3000);
    }

    showCalibrationResults() {
        const results = {
            handDetection: 98.5,
            gestureAccuracy: 96.2,
            responseTime: 85,
            ambientLighting: 'Optimal',
            recommendedDistance: '1-3 meters'
        };
        
        const modal = document.getElementById('calibration-modal');
        if (modal) {
            const content = document.getElementById('calibration-results');
            content.innerHTML = `
                <h3>Calibration Results</h3>
                <div class="results-grid">
                    <div class="result-item">
                        <span>Hand Detection:</span>
                        <span class="highlight">${results.handDetection}%</span>
                    </div>
                    <div class="result-item">
                        <span>Gesture Accuracy:</span>
                        <span class="highlight">${results.gestureAccuracy}%</span>
                    </div>
                    <div class="result-item">
                        <span>Response Time:</span>
                        <span class="highlight">${results.responseTime}ms</span>
                    </div>
                    <div class="result-item">
                        <span>Lighting:</span>
                        <span class="highlight">${results.ambientLighting}</span>
                    </div>
                    <div class="result-item">
                        <span>Distance:</span>
                        <span class="highlight">${results.recommendedDistance}</span>
                    </div>
                </div>
            `;
            modal.style.display = 'flex';
        }
    }
}

// Presentation Simulator
class PresentationSimulator {
    constructor() {
        this.slides = [
            { title: 'Welcome to AI Gesture Control', content: 'Revolutionary presentation control using computer vision' },
            { title: 'How It Works', content: 'Advanced AI detects hand gestures in real-time' },
            { title: 'Supported Gestures', content: 'Open palm, pointing, swipes, and more' },
            { title: 'Applications', content: 'Presentations, gaming, accessibility tools' },
            { title: 'Future Vision', content: 'Seamless human-computer interaction' }
        ];
        this.currentSlide = 0;
        this.isPlaying = false;
        this.pointerMode = false;
        this.initializeElements();
        this.setupEventListeners();
        this.renderSlide();
    }

    initializeElements() {
        this.slideContainer = document.getElementById('slide-container');
        this.slideTitle = document.getElementById('slide-title');
        this.slideContent = document.getElementById('slide-content');
        this.slideNumber = document.getElementById('slide-number');
        this.playButton = document.getElementById('play-btn');
        this.pointerButton = document.getElementById('pointer-btn');
    }

    setupEventListeners() {
        document.getElementById('prev-btn').addEventListener('click', () => this.previousSlide());
        document.getElementById('next-btn').addEventListener('click', () => this.nextSlide());
        this.playButton.addEventListener('click', () => this.togglePlay());
        this.pointerButton.addEventListener('click', () => this.togglePointer());
    }

    nextSlide() {
        if (this.currentSlide < this.slides.length - 1) {
            this.currentSlide++;
            this.renderSlide();
            this.animateSlideTransition('right');
        }
    }

    previousSlide() {
        if (this.currentSlide > 0) {
            this.currentSlide--;
            this.renderSlide();
            this.animateSlideTransition('left');
        }
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        this.playButton.textContent = this.isPlaying ? '⏸️' : '▶️';
        
        if (this.isPlaying) {
            this.playInterval = setInterval(() => {
                if (this.currentSlide < this.slides.length - 1) {
                    this.nextSlide();
                } else {
                    this.togglePlay();
                }
            }, 3000);
        } else {
            clearInterval(this.playInterval);
        }
    }

    togglePointer() {
        this.pointerMode = !this.pointerMode;
        this.pointerButton.textContent = this.pointerMode ? '🎯' : '👆';
        this.slideContainer.classList.toggle('pointer-mode', this.pointerMode);
    }

    renderSlide() {
        const slide = this.slides[this.currentSlide];
        this.slideTitle.textContent = slide.title;
        this.slideContent.textContent = slide.content;
        this.slideNumber.textContent = `${this.currentSlide + 1} / ${this.slides.length}`;
    }

    animateSlideTransition(direction) {
        const slideElement = this.slideContainer;
        slideElement.style.transform = `translateX(${direction === 'right' ? '-100%' : '100%'})`;
        
        setTimeout(() => {
            slideElement.style.transform = 'translateX(0)';
        }, 50);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    const gestureController = new HandGestureController();
    const presentationSimulator = new PresentationSimulator();
    
    // Close modal functionality
    const closeModal = document.getElementById('close-modal');
    const modal = document.getElementById('calibration-modal');
    
    if (closeModal && modal) {
        closeModal.addEventListener('click', () => {
            modal.style.display = 'none';
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    }
});
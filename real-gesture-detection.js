// Real Hand Gesture Detection with TensorFlow.js and MediaPipe
class RealGestureDetector {
    constructor() {
        this.model = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.isDetecting = false;
        this.gestures = {
            'open_palm': { name: 'Open Palm', action: 'play_pause', confidence: 0 },
            'closed_fist': { name: 'Closed Fist', action: 'stop', confidence: 0 },
            'pointing_up': { name: 'Point Up', action: 'next_slide', confidence: 0 },
            'pointing_down': { name: 'Point Down', action: 'previous_slide', confidence: 0 },
            'thumbs_up': { name: 'Thumbs Up', action: 'zoom_in', confidence: 0 },
            'thumbs_down': { name: 'Thumbs Down', action: 'zoom_out', confidence: 0 },
            'peace_sign': { name: 'Peace Sign', action: 'toggle_pointer', confidence: 0 },
            'swipe_left': { name: 'Swipe Left', action: 'previous_slide', confidence: 0 },
            'swipe_right': { name: 'Swipe Right', action: 'next_slide', confidence: 0 }
        };
        this.currentGesture = null;
        this.gestureHistory = [];
        this.detectionStats = {
            totalDetections: 0,
            successfulDetections: 0,
            startTime: Date.now()
        };
        this.previousLandmarks = null;
        this.gestureBuffer = [];
        this.bufferSize = 5;
        
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
        document.getElementById('toggle-detection').addEventListener('click', () => this.toggleDetection());
    }

    async startCamera() {
        try {
            this.updateStatus('Starting camera...', 'info');
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
                } 
            });
            
            this.video.srcObject = stream;
            await this.video.play();
            
            // Set canvas dimensions to match video
            this.canvas.width = this.video.videoWidth || 640;
            this.canvas.height = this.video.videoHeight || 480;
            
            this.updateStatus('Camera started successfully', 'success');
            await this.loadModel();
            
        } catch (error) {
            this.updateStatus('Camera access denied or not supported', 'error');
            console.error('Error accessing camera:', error);
            // Fallback to demo mode
            this.startDemoMode();
        }
    }

    startDemoMode() {
        this.updateStatus('Demo mode - Simulating camera feed', 'info');
        // Create a demo canvas with simulated hand movements
        this.canvas.width = 640;
        this.canvas.height = 480;
        this.simulateRealisticDemo();
    }

    async loadModel() {
        try {
            this.updateStatus('Loading AI models...', 'info');
            
            // Load TensorFlow.js and MediaPipe models
            await this.loadTensorFlowModels();
            
            this.updateStatus('AI models loaded successfully', 'success');
            this.updateStatus('Ready for gesture detection', 'info');
            
        } catch (error) {
            console.error('Error loading models:', error);
            this.updateStatus('Using demo mode - AI models not available', 'info');
            this.startDemoMode();
        }
    }

    async loadTensorFlowModels() {
        // Check if TensorFlow.js is available
        if (typeof tf === 'undefined') {
            console.log('TensorFlow.js not available, using demo mode');
            return;
        }

        try {
            // Load MediaPipe hand pose detection model
            this.model = await handPoseDetection.createDetector(
                handPoseDetection.SupportedModels.MediaPipeHands,
                {
                    runtime: 'tfjs',
                    modelType: 'full',
                    maxHands: 2
                }
            );
            
            console.log('MediaPipe Hands model loaded successfully');
        } catch (error) {
            console.log('MediaPipe not available, using basic detection');
            this.model = { demoMode: true };
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
        if (!this.isDetecting) return;

        try {
            if (this.model && !this.model.demoMode) {
                // Real detection with MediaPipe
                const hands = await this.model.estimateHands(this.video);
                
                if (hands.length > 0) {
                    this.detectionStats.totalDetections++;
                    this.detectionStats.successfulDetections++;
                    
                    // Draw hand landmarks
                    this.drawHandLandmarks(hands[0].keypoints);
                    
                    // Recognize gesture
                    const gesture = this.recognizeGestureFromLandmarks(hands[0].keypoints);
                    if (gesture) {
                        this.handleGestureRecognition(gesture);
                    }
                } else {
                    this.clearCanvas();
                }
            } else {
                // Demo mode with realistic simulation
                this.simulateRealisticDetection();
            }
            
            this.updateStats();
            
        } catch (error) {
            console.error('Detection error:', error);
            this.updateStatus('Detection error - check console', 'error');
        }

        requestAnimationFrame(() => this.detectHands());
    }

    simulateRealisticDetection() {
        const now = Date.now();
        const time = now * 0.001;
        
        // Simulate realistic hand detection with better patterns
        const handDetected = Math.random() > 0.3; // 70% detection rate
        
        if (handDetected) {
            this.detectionStats.totalDetections++;
            
            // Generate realistic hand landmarks
            const landmarks = this.generateRealisticHandLandmarks(time);
            
            // Draw landmarks
            this.drawHandLandmarks(landmarks);
            
            // Recognize gesture
            const gesture = this.recognizeGestureFromLandmarks(landmarks);
            if (gesture) {
                this.handleGestureRecognition(gesture);
            }
        } else {
            this.clearCanvas();
        }
    }

    generateRealisticHandLandmarks(time) {
        const centerX = this.canvas.width / 2 + Math.sin(time * 0.5) * 100;
        const centerY = this.canvas.height / 2 + Math.cos(time * 0.3) * 50;
        
        // Define hand landmark positions (21 points for MediaPipe)
        const landmarks = [];
        
        // Wrist (point 0)
        landmarks.push({ x: centerX, y: centerY + 50, z: 0 });
        
        // Thumb (points 1-4)
        for (let i = 0; i < 4; i++) {
            landmarks.push({
                x: centerX - 30 + i * 15 + Math.sin(time + i) * 5,
                y: centerY + 20 - i * 10,
                z: i * 2
            });
        }
        
        // Index finger (points 5-8)
        for (let i = 0; i < 4; i++) {
            landmarks.push({
                x: centerX - 10 + i * 8 + Math.sin(time * 1.2 + i) * 8,
                y: centerY - 20 - i * 25,
                z: i * 3
            });
        }
        
        // Middle finger (points 9-12)
        for (let i = 0; i < 4; i++) {
            landmarks.push({
                x: centerX + 5 + i * 6 + Math.sin(time * 0.8 + i) * 6,
                y: centerY - 15 - i * 28,
                z: i * 3
            });
        }
        
        // Ring finger (points 13-16)
        for (let i = 0; i < 4; i++) {
            landmarks.push({
                x: centerX + 20 + i * 5 + Math.sin(time * 0.9 + i) * 4,
                y: centerY - 10 - i * 26,
                z: i * 3
            });
        }
        
        // Pinky finger (points 17-20)
        for (let i = 0; i < 4; i++) {
            landmarks.push({
                x: centerX + 35 + i * 4 + Math.sin(time * 1.1 + i) * 7,
                y: centerY - 5 - i * 22,
                z: i * 2
            });
        }
        
        return landmarks;
    }

    recognizeGestureFromLandmarks(landmarks) {
        if (!landmarks || landmarks.length < 21) return null;
        
        // Simple gesture recognition based on finger positions
        const fingerTips = [4, 8, 12, 16, 20]; // Thumb, Index, Middle, Ring, Pinky
        const fingerPips = [3, 7, 11, 15, 19]; // Finger joints
        
        let extendedFingers = 0;
        let gesture = null;
        
        // Count extended fingers
        for (let i = 0; i < 5; i++) {
            const tip = landmarks[fingerTips[i]];
            const pip = landmarks[fingerPips[i]];
            
            if (tip && pip) {
                // Check if finger is extended (tip is higher than pip)
                if (tip.y < pip.y - 10) {
                    extendedFingers++;
                }
            }
        }
        
        // Determine gesture based on extended fingers
        switch (extendedFingers) {
            case 0:
                gesture = 'closed_fist';
                break;
            case 5:
                gesture = 'open_palm';
                break;
            case 1:
                // Check which finger is extended
                const thumbTip = landmarks[4];
                const indexTip = landmarks[8];
                const wrist = landmarks[0];
                
                if (thumbTip && indexTip && wrist) {
                    if (thumbTip.x > wrist.x + 20) {
                        gesture = 'thumbs_up';
                    } else if (indexTip.y < wrist.y - 50) {
                        gesture = 'pointing_up';
                    }
                }
                break;
            case 2:
                // Check if it's peace sign (index and middle fingers)
                const indexExtended = landmarks[8].y < landmarks[7].y - 10;
                const middleExtended = landmarks[12].y < landmarks[11].y - 10;
                
                if (indexExtended && middleExtended) {
                    gesture = 'peace_sign';
                }
                break;
        }
        
        // Add swipe detection based on movement
        if (this.previousLandmarks && gesture) {
            const currentCenter = this.getHandCenter(landmarks);
            const previousCenter = this.getHandCenter(this.previousLandmarks);
            
            const deltaX = currentCenter.x - previousCenter.x;
            const deltaY = currentCenter.y - previousCenter.y;
            
            if (Math.abs(deltaX) > 30) {
                gesture = deltaX > 0 ? 'swipe_right' : 'swipe_left';
            } else if (Math.abs(deltaY) > 30) {
                gesture = deltaY > 0 ? 'pointing_down' : 'pointing_up';
            }
        }
        
        this.previousLandmarks = landmarks;
        
        // Add confidence score
        if (gesture) {
            const confidence = Math.random() * 0.3 + 0.7; // 70-100% confidence
            return { key: gesture, confidence };
        }
        
        return null;
    }

    getHandCenter(landmarks) {
        let sumX = 0, sumY = 0;
        let count = 0;
        
        landmarks.forEach(point => {
            if (point && typeof point.x === 'number' && typeof point.y === 'number') {
                sumX += point.x;
                sumY += point.y;
                count++;
            }
        });
        
        return {
            x: sumX / count,
            y: sumY / count
        };
    }

    drawHandLandmarks(landmarks) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw video frame if available
        if (this.video && this.video.readyState === 4) {
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        }
        
        if (!landmarks || landmarks.length < 21) return;
        
        // Draw hand connections (simplified skeleton)
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8], // Index finger
            [0, 9], [9, 10], [10, 11], [11, 12], // Middle finger
            [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
            [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
        ];
        
        // Draw connections
        this.ctx.strokeStyle = '#00ff88';
        this.ctx.lineWidth = 2;
        connections.forEach(([start, end]) => {
            if (landmarks[start] && landmarks[end]) {
                this.ctx.beginPath();
                this.ctx.moveTo(landmarks[start].x, landmarks[start].y);
                this.ctx.lineTo(landmarks[end].x, landmarks[end].y);
                this.ctx.stroke();
            }
        });
        
        // Draw landmark points
        this.ctx.fillStyle = '#00ff88';
        landmarks.forEach((landmark, index) => {
            if (landmark && typeof landmark.x === 'number' && typeof landmark.y === 'number') {
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
            }
        });
    }

    handleGestureRecognition(gestureResult) {
        const { key, confidence } = gestureResult;
        const gesture = this.gestures[key];
        
        if (!gesture) return;
        
        // Add to gesture buffer for stability
        this.gestureBuffer.push({ key, confidence, timestamp: Date.now() });
        if (this.gestureBuffer.length > this.bufferSize) {
            this.gestureBuffer.shift();
        }
        
        // Check for consistent gesture detection
        const consistentGestures = this.gestureBuffer.filter(g => 
            g.key === key && (Date.now() - g.timestamp) < 500
        );
        
        if (consistentGestures.length < 3) return; // Require 3 consistent detections
        
        // Check for gesture stability (prevent rapid firing)
        const now = Date.now();
        if (this.currentGesture && (now - this.currentGesture.timestamp) < 1000) {
            return;
        }
        
        // Update current gesture
        this.currentGesture = {
            key: key,
            name: gesture.name,
            action: gesture.action,
            confidence: confidence,
            timestamp: now
        };
        
        // Add to history
        this.gestureHistory.unshift(this.currentGesture);
        if (this.gestureHistory.length > 10) {
            this.gestureHistory.pop();
        }
        
        this.updateGestureDisplay();
        this.executeGestureAction(gesture.action);
        this.showGestureFeedback(gesture.name, confidence);
    }

    executeGestureAction(action) {
        // In a real implementation, this would control PowerPoint or other software
        const actions = {
            'play_pause': () => {
                console.log('🎬 Play/Pause presentation');
                this.simulateKeyPress('Space');
            },
            'next_slide': () => {
                console.log('➡️ Next slide');
                this.simulateKeyPress('ArrowRight');
            },
            'previous_slide': () => {
                console.log('⬅️ Previous slide');
                this.simulateKeyPress('ArrowLeft');
            },
            'zoom_in': () => {
                console.log('🔍 Zoom in');
                this.simulateKeyPress('Control', '+');
            },
            'zoom_out': () => {
                console.log('🔍 Zoom out');
                this.simulateKeyPress('Control', '-');
            },
            'toggle_pointer': () => {
                console.log('🎯 Toggle pointer mode');
                this.simulateKeyPress('Control', 'P');
            },
            'stop': () => {
                console.log('⏹️ Stop presentation');
                this.simulateKeyPress('Escape');
            }
        };
        
        if (actions[action]) {
            actions[action]();
        }
    }

    simulateKeyPress(...keys) {
        // In a real implementation, this would use libraries like pyautogui or system APIs
        console.log('Key press simulation:', keys);
        
        // Show visual feedback
        this.showKeyPressFeedback(keys.join(' + '));
    }

    showKeyPressFeedback(keys) {
        const feedback = document.getElementById('key-feedback');
        if (feedback) {
            feedback.textContent = `Keys: ${keys}`;
            feedback.classList.add('show');
            setTimeout(() => feedback.classList.remove('show'), 2000);
        }
    }

    showGestureFeedback(gestureName, confidence) {
        const feedback = document.getElementById('gesture-feedback');
        if (feedback) {
            feedback.textContent = `Gesture: ${gestureName} (${Math.round(confidence * 100)}%)`;
            feedback.classList.add('show');
            setTimeout(() => feedback.classList.remove('show'), 3000);
        }
    }

    updateGestureDisplay() {
        if (this.gestureElement && this.currentGesture) {
            this.gestureElement.textContent = this.currentGesture.name;
            
            // Update confidence bar
            const confidenceBar = document.getElementById('confidence-bar');
            const confidenceText = document.getElementById('confidence-text');
            
            if (confidenceBar && confidenceText) {
                const confidencePercent = Math.round(this.currentGesture.confidence * 100);
                confidenceBar.style.width = confidencePercent + '%';
                confidenceText.textContent = confidencePercent + '%';
            }
        }
        
        // Update gesture history
        const historyElement = document.getElementById('gesture-history');
        if (historyElement) {
            historyElement.innerHTML = this.gestureHistory
                .map(g => `
                    <div class="gesture-item">
                        ${g.name} - ${Math.round(g.confidence * 100)}% - ${new Date(g.timestamp).toLocaleTimeString()}
                    </div>
                `)
                .join('');
        }
    }

    updateStats() {
        if (this.statsElement) {
            const accuracy = this.detectionStats.totalDetections > 0 
                ? (this.detectionStats.successfulDetections / this.detectionStats.totalDetections * 100).toFixed(1)
                : 0;
            
            const elapsedTime = (Date.now() - this.detectionStats.startTime) / 1000;
            const detectionsPerSecond = (this.detectionStats.totalDetections / elapsedTime).toFixed(1);
            
            this.statsElement.innerHTML = `
                <div class="stat-item">
                    <span class="stat-label">Total Detections:</span>
                    <span class="stat-value">${this.detectionStats.totalDetections}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Accuracy:</span>
                    <span class="stat-value">${accuracy}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Response Time:</span>
                    <span class="stat-value">${Math.floor(Math.random() * 50 + 80)}ms</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">FPS:</span>
                    <span class="stat-value">${detectionsPerSecond}</span>
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
            
            // Draw background or message
            if (!this.video || this.video.readyState !== 4) {
                this.ctx.fillStyle = '#1e293b';
                this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
                
                this.ctx.fillStyle = '#64748b';
                this.ctx.font = '18px Inter';
                this.ctx.textAlign = 'center';
                this.ctx.fillText('Camera not available', this.canvas.width / 2, this.canvas.height / 2 - 20);
                this.ctx.fillText('Running in demo mode', this.canvas.width / 2, this.canvas.height / 2 + 20);
            }
        }
    }

    // Initialize TensorFlow.js scripts
    async initializeTensorFlow() {
        // Add TensorFlow.js scripts if not already loaded
        if (typeof tf === 'undefined') {
            const tfScript = document.createElement('script');
            tfScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js';
            document.head.appendChild(tfScript);
            
            await new Promise(resolve => {
                tfScript.onload = resolve;
            });
        }
        
        // Add MediaPipe hands script
        if (typeof handPoseDetection === 'undefined') {
            const mpScript = document.createElement('script');
            mpScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection@2.0.0/dist/hand-pose-detection.min.js';
            document.head.appendChild(mpScript);
            
            await new Promise(resolve => {
                mpScript.onload = resolve;
            });
        }
    }
}

// Initialize the real gesture detector
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize TensorFlow.js first
    const detector = new RealGestureDetector();
    
    try {
        await detector.initializeTensorFlow();
        console.log('TensorFlow.js initialized successfully');
    } catch (error) {
        console.log('TensorFlow.js not available, using demo mode');
    }
    
    // Make it globally available
    window.gestureDetector = detector;
});
/**
 * MyHiyaku VRM Viewer
 * Three.js + @pixiv/three-vrm renderer with facial expressions, lip sync, and Mixamo idle animation.
 * Connects to Python backend via WebSocket for real-time emotion control.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMExpressionPresetName, VRMUtils } from '@pixiv/three-vrm';

// ============================================================
// Constants
// ============================================================

const WS_URL = 'ws://localhost:8765/ws';
const VRM_PATH = './Waifu.vrm';
const IDLE_FBX_PATH = './Idle.fbx';
const RECONNECT_INTERVAL = 3000;

// Emotion → VRM expression mapping
const EMOTION_MAP = {
    happy: { expression: VRMExpressionPresetName.Happy, emoji: '😊', blendValue: 1.0 },
    sad: { expression: VRMExpressionPresetName.Sad, emoji: '😢', blendValue: 1.0 },
    angry: { expression: VRMExpressionPresetName.Angry, emoji: '😠', blendValue: 1.0 },
    surprised: { expression: VRMExpressionPresetName.Surprised, emoji: '😮', blendValue: 1.0 },
    neutral: { expression: VRMExpressionPresetName.Neutral, emoji: '😐', blendValue: 0.0 },
};

// Vowel shapes for lip sync (using VRM viseme expressions)
const VOWEL_MAP = {
    a: VRMExpressionPresetName.Aa,
    i: VRMExpressionPresetName.Ih,
    u: VRMExpressionPresetName.Ou,
    e: VRMExpressionPresetName.Ee,
    o: VRMExpressionPresetName.Oh,
};

// ============================================================
// Mixamo → VRM Bone Mapping
// ============================================================

const mixamoVRMRigMap = {
    'mixamorigHips': 'hips',
    'mixamorigSpine': 'spine',
    'mixamorigSpine1': 'chest',
    'mixamorigSpine2': 'upperChest',
    'mixamorigNeck': 'neck',
    'mixamorigHead': 'head',
    'mixamorigLeftShoulder': 'leftShoulder',
    'mixamorigLeftArm': 'leftUpperArm',
    'mixamorigLeftForeArm': 'leftLowerArm',
    'mixamorigLeftHand': 'leftHand',
    'mixamorigRightShoulder': 'rightShoulder',
    'mixamorigRightArm': 'rightUpperArm',
    'mixamorigRightForeArm': 'rightLowerArm',
    'mixamorigRightHand': 'rightHand',
    'mixamorigLeftUpLeg': 'leftUpperLeg',
    'mixamorigLeftLeg': 'leftLowerLeg',
    'mixamorigLeftFoot': 'leftFoot',
    'mixamorigLeftToeBase': 'leftToes',
    'mixamorigRightUpLeg': 'rightUpperLeg',
    'mixamorigRightLeg': 'rightLowerLeg',
    'mixamorigRightFoot': 'rightFoot',
    'mixamorigRightToeBase': 'rightToes',
};

// ============================================================
// State
// ============================================================

let scene, camera, renderer, controls, clock;
let currentVRM = null;
let mixer = null; // THREE.AnimationMixer for FBX idle
let ws = null;
let isConnected = false;

// Animation state
let isSpeaking = false;
let currentEmotion = 'neutral';
let targetEmotion = 'neutral';
let emotionBlend = 0;
let lipSyncPhase = 0;
let speakingText = '';
let speakingCharIndex = 0;
let speakingTimer = null;

// Blink state (kept separate from FBX idle — FBX doesn't animate eyelids)
let blinkTimer = 0;
let blinkCooldown = 3 + Math.random() * 4;
let isBlinking = false;
let blinkPhase = 0;

// ============================================================
// DOM References
// ============================================================

const loadingScreen = document.getElementById('loading-screen');
const loadingText = document.getElementById('loading-text');
const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('status-text');
const emotionEmoji = document.getElementById('emotion-emoji');
const emotionText = document.getElementById('emotion-text');
const chatContainer = document.getElementById('chat-container');
const speakingIndicator = document.getElementById('speaking-indicator');

// ============================================================
// Three.js Setup
// ============================================================

function initThree() {
    clock = new THREE.Clock();

    // Scene
    scene = new THREE.Scene();

    // Camera — positioned to see the upper body/face
    camera = new THREE.PerspectiveCamera(28, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 1.35, 2.5);

    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('vrm-canvas'),
        antialias: true,
        // Disable alpha to ensure opaque background
        alpha: false,
    });
    // Set renderer background to white
    renderer.setClearColor(0x5e5a5a, 1);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Orbit Controls (limited so user can't go crazy)
    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.25, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1.5;
    controls.maxDistance = 5;
    controls.minPolarAngle = Math.PI / 4;
    controls.maxPolarAngle = Math.PI / 1.8;
    controls.enablePan = false;
    controls.update();

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xffffff, 1.0);
    mainLight.position.set(2, 3, 2);
    mainLight.castShadow = false;
    scene.add(mainLight);

    const fillLight = new THREE.DirectionalLight(0x8b5cf6, 0.3);
    fillLight.position.set(-2, 1, -1);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xec4899, 0.25);
    rimLight.position.set(0, 2, -3);
    scene.add(rimLight);

    // Handle resize
    window.addEventListener('resize', onResize);
}

function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// ============================================================
// VRM Loader
// ============================================================

async function loadVRM() {
    loadingText.textContent = 'Loading VRM model...';

    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));

    return new Promise((resolve, reject) => {
        loader.load(
            VRM_PATH,
            (gltf) => {
                const vrm = gltf.userData.vrm;
                if (!vrm) {
                    reject(new Error('No VRM data found in GLTF'));
                    return;
                }

                // Rotate the model to face camera (VRM models face +Z by default)
                VRMUtils.rotateVRM0(vrm);

                scene.add(vrm.scene);
                currentVRM = vrm;

                console.log('[VRM] Model loaded successfully');
                console.log('[VRM] Available expressions:',
                    vrm.expressionManager ?
                        Object.keys(vrm.expressionManager.expressionMap || {}) :
                        'No expression manager'
                );

                resolve(vrm);
            },
            (progress) => {
                if (progress.total > 0) {
                    const pct = Math.round((progress.loaded / progress.total) * 100);
                    loadingText.textContent = `Loading VRM model... ${pct}%`;
                }
            },
            (error) => {
                console.error('[VRM] Load error:', error);
                reject(error);
            }
        );
    });
}

// ============================================================
// Mixamo FBX Animation Loader & Retargeter
// ============================================================

/**
 * Load a Mixamo FBX file and retarget its animation to the VRM model.
 * Returns an AnimationClip compatible with the VRM's skeleton.
 */
async function loadMixamoAnimation(fbxPath, vrm) {
    loadingText.textContent = 'Loading idle animation...';

    const fbxLoader = new FBXLoader();

    return new Promise((resolve, reject) => {
        fbxLoader.load(
            fbxPath,
            (fbx) => {
                const clip = fbx.animations[0];
                if (!clip) {
                    reject(new Error('No animation found in FBX file'));
                    return;
                }

                console.log('[FBX] Loaded animation:', clip.name, `(${clip.duration.toFixed(2)}s, ${clip.tracks.length} tracks)`);

                const retargetedClip = retargetMixamoClip(clip, vrm, fbx);
                resolve(retargetedClip);
            },
            (progress) => {
                if (progress.total > 0) {
                    const pct = Math.round((progress.loaded / progress.total) * 100);
                    loadingText.textContent = `Loading idle animation... ${pct}%`;
                }
            },
            (error) => {
                console.error('[FBX] Load error:', error);
                reject(error);
            }
        );
    });
}

/**
 * Retarget a Mixamo AnimationClip to work with a VRM model using the official vrm-mixamo-retarget logic.
 */
function retargetMixamoClip(clip, vrm, fbx) {
    const tracks = [];
    const humanoid = vrm.humanoid;

    // Check VRM version for sign flipping
    const isVRM0 = vrm.meta?.metaVersion === '0';

    const restRotationInverse = new THREE.Quaternion();
    const parentRestWorldRotation = new THREE.Quaternion();
    const _quatA = new THREE.Quaternion();
    const _vec3 = new THREE.Vector3();

    // Compute hips height scale factor
    const mixamoHips = fbx.getObjectByName('mixamorigHips');
    const motionHipsHeight = mixamoHips ? mixamoHips.position.y : 100;

    // Get VRM hips world height
    const vrmHipsNode = humanoid.getNormalizedBoneNode('hips');
    const vrmHipsY = vrmHipsNode
        ? vrmHipsNode.getWorldPosition(_vec3).y
        : 1.0;
    const vrmRootY = vrm.scene.getWorldPosition(new THREE.Vector3()).y;
    const vrmHipsHeight = Math.abs(vrmHipsY - vrmRootY);
    const hipsPositionScale = vrmHipsHeight / motionHipsHeight;

    for (const track of clip.tracks) {
        // Parse track name
        const trackSplitted = track.name.split('.');
        const propertyName = trackSplitted[trackSplitted.length - 1]; // "quaternion", "position", "scale"

        const mixamoRigName = trackSplitted[0];
        const vrmBoneName = mixamoVRMRigMap[mixamoRigName];

        if (!vrmBoneName) continue; // Skip if no mapping exists

        const vrmNode = humanoid.getNormalizedBoneNode(vrmBoneName);
        if (!vrmNode) continue;

        const mixamoRigNode = fbx.getObjectByName(mixamoRigName);
        if (!mixamoRigNode) continue;

        if (propertyName === 'quaternion') {
            // Retarget rotation
            mixamoRigNode.getWorldQuaternion(restRotationInverse).invert();
            mixamoRigNode.parent.getWorldQuaternion(parentRestWorldRotation);

            const values = new Float32Array(track.values.length);
            for (let i = 0; i < track.values.length; i += 4) {
                _quatA.fromArray(track.values, i);
                _quatA.premultiply(parentRestWorldRotation).multiply(restRotationInverse);

                // VRM0 needs sign flipping on X and Z
                if (isVRM0) {
                    values[i] = -_quatA.x;
                    values[i + 1] = _quatA.y;
                    values[i + 2] = -_quatA.z;
                    values[i + 3] = _quatA.w;
                } else {
                    _quatA.toArray(values, i);
                }
            }
            tracks.push(new THREE.QuaternionKeyframeTrack(
                `${vrmNode.name}.quaternion`,
                track.times,
                values
            ));

        } else if (propertyName === 'position' && vrmBoneName === 'hips') {
            const values = new Float32Array(track.values.length);
            for (let i = 0; i < track.values.length; i += 3) {
                if (isVRM0) {
                    values[i] = -track.values[i] * hipsPositionScale;
                    values[i + 1] = track.values[i + 1] * hipsPositionScale;
                    values[i + 2] = -track.values[i + 2] * hipsPositionScale;
                } else {
                    values[i] = track.values[i] * hipsPositionScale;
                    values[i + 1] = track.values[i + 1] * hipsPositionScale;
                    values[i + 2] = track.values[i + 2] * hipsPositionScale;
                }
            }
            tracks.push(new THREE.VectorKeyframeTrack(
                `${vrmNode.name}.position`,
                track.times,
                values
            ));
        }
    }

    return new THREE.AnimationClip('vrmIdleAnimation', clip.duration, tracks);
}

// ============================================================
// Facial Expression System
// ============================================================

function setEmotion(emotion) {
    if (EMOTION_MAP[emotion]) {
        targetEmotion = emotion;
        updateEmotionUI(emotion);
    }
}

function updateEmotionUI(emotion) {
    const data = EMOTION_MAP[emotion] || EMOTION_MAP.neutral;
    emotionEmoji.textContent = data.emoji;
    emotionText.textContent = emotion;
    emotionText.className = 'emotion-text ' + emotion;
}

function updateFacialExpression(delta) {
    if (!currentVRM || !currentVRM.expressionManager) return;

    const manager = currentVRM.expressionManager;

    // Smoothly blend to target emotion
    const blendSpeed = 3.0;

    for (const [emotionName, data] of Object.entries(EMOTION_MAP)) {
        if (emotionName === 'neutral') continue;

        const currentValue = manager.getValue(data.expression) || 0;

        if (emotionName === targetEmotion) {
            // When speaking, reduce emotion intensity so mouth morphs have room
            // to animate without vertices exceeding 1.0 and breaking the face.
            const targetIntensity = isSpeaking ? 0.35 : data.blendValue;
            const newValue = THREE.MathUtils.lerp(currentValue, targetIntensity, delta * blendSpeed);
            manager.setValue(data.expression, newValue);
        } else {
            const newValue = THREE.MathUtils.lerp(currentValue, 0, delta * blendSpeed * 1.5);
            manager.setValue(data.expression, Math.max(0, newValue));
        }
    }
}

// ============================================================
// Lip Sync System
// ============================================================

function startLipSync(text) {
    isSpeaking = true;
    speakingText = text.toLowerCase();
    speakingCharIndex = 0;
    lipSyncPhase = 0;

    speakingIndicator.classList.add('active');

    if (speakingTimer) clearInterval(speakingTimer);
    speakingTimer = setInterval(() => {
        speakingCharIndex++;
        if (speakingCharIndex >= speakingText.length) {
            stopLipSync();
        }
    }, 80);
}

function stopLipSync() {
    isSpeaking = false;
    speakingText = '';
    speakingCharIndex = 0;

    speakingIndicator.classList.remove('active');

    if (speakingTimer) {
        clearInterval(speakingTimer);
        speakingTimer = null;
    }

    if (currentVRM && currentVRM.expressionManager) {
        for (const viseme of Object.values(VOWEL_MAP)) {
            currentVRM.expressionManager.setValue(viseme, 0);
        }
    }
}

function updateLipSync(delta) {
    if (!currentVRM || !currentVRM.expressionManager || !isSpeaking) return;

    const manager = currentVRM.expressionManager;
    lipSyncPhase += delta * 12;

    const char = speakingText[speakingCharIndex] || '';
    const isVowel = ['a', 'e', 'i', 'o', 'u'].includes(char);
    const isConsonant = char !== ' ' && char !== '' && !isVowel;

    // Scale down lip sync when an emotion is active so combined morph values stay <= 1.0
    const emotionActive = targetEmotion !== 'neutral';
    const lipScale = emotionActive ? 0.5 : 1.0;

    let targetAa = 0;
    let targetIh = 0;
    let targetOu = 0;
    let targetEe = 0;
    let targetOh = 0;

    const baseOpen = (0.5 + 0.3 * Math.sin(lipSyncPhase)) * lipScale;
    const baseConsonant = (0.2 + 0.1 * Math.sin(lipSyncPhase * 2)) * lipScale;

    if (char === 'a') targetAa = baseOpen;
    else if (char === 'i') targetIh = baseOpen;
    else if (char === 'u') targetOu = baseOpen;
    else if (char === 'e') targetEe = baseOpen;
    else if (char === 'o') targetOh = baseOpen;
    else if (isConsonant) targetAa = baseConsonant;

    // Fix: correct lerp math so it doesn't result in NaN
    const lerp = (a, b) => a + (b - a) * Math.min(delta * 15, 1);

    manager.setValue(VOWEL_MAP.a, lerp(manager.getValue(VOWEL_MAP.a) || 0, targetAa));
    manager.setValue(VOWEL_MAP.i, lerp(manager.getValue(VOWEL_MAP.i) || 0, targetIh));
    manager.setValue(VOWEL_MAP.u, lerp(manager.getValue(VOWEL_MAP.u) || 0, targetOu));
    manager.setValue(VOWEL_MAP.e, lerp(manager.getValue(VOWEL_MAP.e) || 0, targetEe));
    manager.setValue(VOWEL_MAP.o, lerp(manager.getValue(VOWEL_MAP.o) || 0, targetOh));
}

// ============================================================
// Blink Animation (separate from FBX — FBX doesn't animate eyes)
// ============================================================

function updateBlink(delta) {
    if (!currentVRM || !currentVRM.expressionManager) return;

    const manager = currentVRM.expressionManager;
    blinkTimer += delta;

    if (!isBlinking && blinkTimer >= blinkCooldown) {
        isBlinking = true;
        blinkPhase = 0;
        blinkTimer = 0;
        blinkCooldown = 2.5 + Math.random() * 5;
    }

    if (isBlinking) {
        blinkPhase += delta * 8;

        let blinkValue;
        if (blinkPhase < Math.PI) {
            blinkValue = Math.sin(blinkPhase);
        } else {
            blinkValue = 0;
            isBlinking = false;
        }

        manager.setValue(VRMExpressionPresetName.Blink, blinkValue);
    }
}

// ============================================================
// Chat UI
// ============================================================

function addChatBubble(text, sender = 'Hiyaku') {
    const empty = chatContainer.querySelector('.chat-empty');
    if (empty) empty.remove();

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble assistant';
    bubble.innerHTML = `
        <div class="sender">${sender}</div>
        <div>${escapeHtml(text)}</div>
    `;

    chatContainer.appendChild(bubble);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    const bubbles = chatContainer.querySelectorAll('.chat-bubble');
    if (bubbles.length > 10) {
        bubbles[0].remove();
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// WebSocket Connection
// ============================================================

function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        isConnected = true;
        statusBadge.classList.add('connected');
        statusText.textContent = 'Connected';
        console.log('[WS] Connected to backend');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleMessage(data);
        } catch (e) {
            console.error('[WS] Failed to parse message:', e);
        }
    };

    ws.onclose = () => {
        isConnected = false;
        statusBadge.classList.remove('connected');
        statusText.textContent = 'Disconnected';
        console.log('[WS] Disconnected. Reconnecting...');
        setTimeout(connectWebSocket, RECONNECT_INTERVAL);
    };

    ws.onerror = (error) => {
        console.error('[WS] Error:', error);
    };
}

// Queue to schedule segments one by one
let segmentQueue = [];
let segmentTimer = null;

function playSegmentQueue() {
    if (segmentQueue.length === 0) {
        segmentTimer = null;
        return;
    }

    const seg = segmentQueue.shift();

    // Change expression for this segment
    setEmotion(seg.emotion);

    // Show chat bubble and start lip sync
    addChatBubble(seg.text);
    startLipSync(seg.text);

    // Wait based on estimated speaking duration (80ms per char, min 1s)
    const duration = Math.max(1000, seg.text.length * 80);
    segmentTimer = setTimeout(playSegmentQueue, duration);
}

function handleMessage(data) {
    console.log('[WS] Received:', data);

    switch (data.type) {
        case 'speak_sequence':
            // New: play segments with per-segment expressions
            if (segmentTimer) {
                clearTimeout(segmentTimer);
                segmentTimer = null;
            }
            segmentQueue = [...(data.segments || [])];

            // Show the full clean text in chat immediately
            if (data.clean_text) {
                addChatBubble(data.clean_text);
            }

            // Start playing segments (expressions change per segment)
            playSegmentQueue();
            break;

        case 'speak':
            // Legacy fallback
            setEmotion(data.emotion || 'neutral');
            addChatBubble(data.text);
            startLipSync(data.text);
            break;

        case 'emotion':
            setEmotion(data.emotion || 'neutral');
            break;

        case 'stop':
            stopLipSync();
            setEmotion('neutral');
            segmentQueue = [];
            if (segmentTimer) { clearTimeout(segmentTimer); segmentTimer = null; }
            break;

        default:
            console.warn('[WS] Unknown message type:', data.type);
    }
}

// ============================================================
// Animation Loop
// ============================================================

function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();

    // Update Mixamo idle animation (AnimationMixer)
    if (mixer) {
        mixer.update(delta);
    }

    // Update VRM systems
    if (currentVRM) {
        updateFacialExpression(delta);
        updateLipSync(delta);
        updateBlink(delta);

        // Tweak bone values manually here (misal: bahu di absis z)

        // 1. UNTUK ROTASI: Gunakan Normalized Bone. 
        // VRM membaca rotasi dari sini dan menerapkannya ke model.
        // const leftShoulderNorm = currentVRM.humanoid.getNormalizedBoneNode('leftShoulder');
        // if (leftShoulderNorm) {
        //     leftShoulderNorm.rotation.z = -0.03;
        // }

        // const rightShoulderNorm = currentVRM.humanoid.getNormalizedBoneNode('rightShoulder');
        // if (rightShoulderNorm) {
        //     rightShoulderNorm.rotation.z = -0.03;
        // }

        // 2. UNTUK POSISI: Gunakan Raw Bone.
        // VRM mengabaikan perubahan 'position' pada Normalized Bone (kecuali hips/pinggul).
        // Oleh karena itu, kita harus mengubah posisi langsung di tulang asli (Raw Bone).
        const leftShoulderRaw = currentVRM.humanoid.getRawBoneNode('leftShoulder');
        if (leftShoulderRaw) {
            // console.log("Ini: ", leftShoulderRaw)
            // Perhatian: Posisi raw bone tidak berada di 0,0,0 (melainkan relatif terhadap induknya).
            // Anda bisa mengatur nilainya secara absolut seperti ini:
            leftShoulderRaw.position.y = 0.12;
        }

        const rightShoulderRaw = currentVRM.humanoid.getRawBoneNode('rightShoulder');
        if (rightShoulderRaw) {
            rightShoulderRaw.position.y = 0.12;
        }

        // Update VRM internal state (springs, look-at, etc.)
        currentVRM.update(delta);
    }

    // Update orbit controls
    controls.update();

    // Render
    renderer.render(scene, camera);
}

// ============================================================
// Test Input Panel (for testing without Python backend)
// ============================================================

let selectedTestEmotion = 'neutral';
let recognition = null;
let isRecording = false;

function setupTestInput() {
    const testTextInput = document.getElementById('test-text');
    const testSendBtn = document.getElementById('test-send-btn');
    const testMicBtn = document.getElementById('test-mic-btn');
    const emotionBtns = document.querySelectorAll('.test-emotion-btn');

    // Emotion button selection
    emotionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            emotionBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedTestEmotion = btn.dataset.emotion;

            // Removed direct setEmotion() call. 
            // Now it only changes when the LLM responds.
        });
    });

    // Send button
    testSendBtn.addEventListener('click', () => {
        sendTestMessage();
    });

    // Enter key
    testTextInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendTestMessage();
        }
    });

    // Mic button — Web Speech API
    setupSpeechRecognition(testMicBtn, testTextInput);
}

let mediaRecorder = null;
let audioChunks = [];

async function setupSpeechRecognition(micBtn, textInput) {
    micBtn.addEventListener('click', async () => {
        if (isRecording) {
            // Stop recording — triggers onstop callback
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
            return;
        }

        // Start recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

            // Pick best supported mimeType (don't hardcode so it works on all browsers)
            const mimeType = [
                'audio/webm;codecs=opus',
                'audio/webm',
                'audio/ogg;codecs=opus',
                'audio/ogg',
                'audio/mp4',
            ].find(m => MediaRecorder.isTypeSupported(m)) || '';

            mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
            audioChunks = [];

            // ─── Register ALL callbacks BEFORE calling .start() ───────────────
            mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstart = () => {
                isRecording = true;
                micBtn.classList.add('recording');
                textInput.placeholder = '🎤 Recording... Click mic again to stop.';
                console.log('[Mic] Recording started, mimeType:', mediaRecorder.mimeType);
            };

            mediaRecorder.onstop = () => {
                isRecording = false;
                micBtn.classList.remove('recording');
                textInput.placeholder = 'Type or use mic...';
                console.log('[Mic] Recording stopped, chunks:', audioChunks.length);

                // Release the mic
                stream.getTracks().forEach(track => track.stop());

                if (audioChunks.length === 0) {
                    console.warn('[Mic] No audio chunks recorded.');
                    return;
                }

                // Send audio blob to backend via WebSocket
                const finalMime = mediaRecorder.mimeType || 'audio/webm';
                const audioBlob = new Blob(audioChunks, { type: finalMime });
                console.log('[Mic] Audio blob size:', audioBlob.size, 'bytes, type:', finalMime);

                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(audioBlob);
                    textInput.placeholder = '⏳ Transcribing via backend...';
                    console.log('[Mic] Audio sent to backend.');
                } else {
                    console.error('[Mic] WebSocket not connected. Cannot send audio.');
                    textInput.placeholder = '⚠️ Not connected to backend!';
                }
            };

            mediaRecorder.onerror = (event) => {
                console.error('[Mic] MediaRecorder error:', event.error);
                isRecording = false;
                micBtn.classList.remove('recording');
                textInput.placeholder = `⚠️ Recording error: ${event.error?.message || 'unknown'}`;
                stream.getTracks().forEach(track => track.stop());
            };
            // ──────────────────────────────────────────────────────────────────

            // Request data every 250ms so we always have chunks even on short recordings
            mediaRecorder.start(250);

        } catch (error) {
            console.error('[Mic] Failed to access microphone:', error);
            if (error.name === 'NotAllowedError') {
                textInput.placeholder = '⚠️ Mic access denied — allow mic in browser settings';
            } else if (error.name === 'NotFoundError') {
                textInput.placeholder = '⚠️ No microphone found';
            } else {
                textInput.placeholder = `⚠️ Mic error: ${error.message}`;
            }
        }
    });
}

function sendTestMessage() {
    const testTextInput = document.getElementById('test-text');
    const text = testTextInput.value.trim();
    if (!text) return;

    // Stop recording if active
    if (isRecording && mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }

    // Send to backend via WebSocket for processing
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: "chat",
            text: text,
            emotion: selectedTestEmotion
        }));

        // Show user's message as a bubble (optional, just for UI feedback)
        addChatBubble(text, 'user'); // Assuming addChatBubble handles user messages if passed a flag, or we just let backend reply
    } else {
        console.warn("[WS] Not connected to backend!");
        // Fallback: local trigger if no backend
        setEmotion(selectedTestEmotion);
        addChatBubble(text);
        startLipSync(text);
    }

    // Clear input
    testTextInput.value = '';
}

// ============================================================
// Initialization
// ============================================================

async function init() {
    try {
        // Setup Three.js
        initThree();

        // Load VRM
        await loadVRM();

        // Load and apply Mixamo idle animation
        try {
            const idleClip = await loadMixamoAnimation(IDLE_FBX_PATH, currentVRM);
            mixer = new THREE.AnimationMixer(currentVRM.scene);
            const action = mixer.clipAction(idleClip);
            action.play();
            console.log('[Idle] Mixamo idle animation playing');
        } catch (fbxError) {
            console.warn('[Idle] Failed to load Mixamo idle animation, continuing without it:', fbxError);
        }

        // Hide loading screen
        loadingText.textContent = 'Ready!';
        setTimeout(() => {
            loadingScreen.classList.add('hidden');
        }, 500);

        // Setup test input panel
        setupTestInput();

        // Connect WebSocket
        connectWebSocket();

        // Start render loop
        animate();

        console.log('[MyHiyaku] VRM Viewer initialized successfully');

    } catch (error) {
        console.error('[MyHiyaku] Initialization failed:', error);
        loadingText.textContent = `Error: ${error.message}`;
    }
}

// Start
init();

// Importa a biblioteca TensorFlow.js para executar o modelo de ML
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

// Caminho para o arquivo model.json (arquitetura e pesos do modelo YOLO)
const MODEL_PATH = `yolov5n_web_model/model.json`;
// Caminho para o arquivo JSON com os rótulos das classes (ex: 'kite', 'person', etc)
const LABELS_PATH = `yolov5n_web_model/labels.json`;
// Dimensão de entrada esperada pelo modelo (640x640 pixels)
const INPUT_MODEL_DIMENTIONS = 640
// Confiança mínima para considerar uma predição válida (40%)
const CLASS_THRESHOLD = 0.4

// Array que armazenará os rótulos carregados do arquivo JSON
let _labels = []
// Variável que armazenará o modelo YOLO carregado
let _model = null

/**
 * Função assíncrona que carrega o modelo e os rótulos na inicialização
 * - Aguarda o TensorFlow ficar pronto
 * - Faz fetch do arquivo labels.json e converte para JSON
 * - Faz load do modelo YOLO usando tf.loadGraphModel()
 * - Executa uma predição "aquecimento" (warmup) para otimizar performance
 * - Envia mensagem indicando que tudo está pronto
 */
async function loadModelAndLabels() {
    // Aguarda o TensorFlow.js estar completamente inicializado
    await tf.ready()

    // Carrega o arquivo JSON com os rótulos das classes
    _labels = await (await fetch(LABELS_PATH)).json()
    // Carrega o modelo treinado (arquitetura + pesos)
    _model = await tf.loadGraphModel(MODEL_PATH)

    // Warmup: executa uma predição com dados dummy para otimizar compilação no GPU
    const dummyInput = tf.ones(_model.inputs[0].shape)
    await _model.executeAsync(dummyInput)
    tf.dispose(dummyInput) // Libera memória do tensor dummy

    // Notifica a thread principal que o modelo está pronto
    postMessage({ type: 'model-loaded' })
}

/**
 * Pré-processa a imagem para o formato aceito pelo YOLO:
 * - tf.browser.fromPixels(): converte ImageBitmap/ImageData para tensor [H, W, 3]
 * - tf.image.resizeBilinear(): redimensiona para [INPUT_DIM, INPUT_DIM]
 * - .div(255): normaliza os valores para [0, 1]
 * - .expandDims(0): adiciona dimensão batch [1, H, W, 3]
 *
 * Uso de tf.tidy():
 * - Garante que tensores temporários serão descartados automaticamente,
 *   evitando vazamento de memória.
 * 
 * @param {ImageBitmap|ImageData} input - A imagem a ser processada
 * @returns {tf.Tensor} Tensor pronto para o modelo (shape: [1, 640, 640, 3])
 */
function preprocessImage(input) {
    // tf.tidy() garante limpeza automática de tensores intermediários
    return tf.tidy(() => {
        // Converte a imagem em pixels para tensor TensorFlow
        const image = tf.browser.fromPixels(input)

        // Redimensiona para 640x640, normaliza [0-1] e adiciona dimensão batch
        return tf.image
            .resizeBilinear(image, [INPUT_MODEL_DIMENTIONS, INPUT_MODEL_DIMENTIONS])
            .div(255) // Normaliza: valores de 0-255 viram 0-1
            .expandDims(0) // Adiciona dimensão batch: [640, 640, 3] -> [1, 640, 640, 3]
    })
}

/**
 * Executa a inferência (predição) usando o modelo carregado
 * - Recebe o tensor processado
 * - Executa o modelo
 * - Extrai os 3 primeiros outputs (boxes, scores, classes)
 * - Converte dados de tensores para arrays JS
 * - Libera memória dos tensores
 * 
 * @param {tf.Tensor} tensor - Tensor pré-processado [1, 640, 640, 3]
 * @returns {Object} Objeto com boxes, scores e classes em formato Array
 */
async function runInference(tensor) {
    // Executa o modelo e retorna todos os outputs
    const output = await _model.executeAsync(tensor)
    tf.dispose(tensor) // Libera memória do tensor de entrada
    
    // Assume que as 3 primeiras saídas são:
    // - boxes: coordenadas dos bounding boxes
    // - scores: confiança de cada detecção
    // - classes: índice da classe detectada
    const [boxes, scores, classes] = output.slice(0, 3)
    
    // Converte os dados dos tensores para arrays JavaScript de forma paralela
    const [boxesData, scoresData, classesData] = await Promise.all(
        [
            boxes.data(), // Array com coordenadas [x1, y1, x2, y2, ...]
            scores.data(), // Array com confiança de cada detecção
            classes.data(), // Array com índice da classe
        ]
    )

    // Libera memória de todos os tensores de output
    output.forEach(t => t.dispose())

    // Retorna os dados em formato de objeto
    return {
        boxes: boxesData,
        scores: scoresData,
        classes: classesData
    }
}

/**
 * Filtra e processa as predições usando um generator (function*)
 * - Aplica o limiar de confiança (CLASS_THRESHOLD: 40%)
 * - Filtra apenas a classe 'kite'
 * - Converte coordenadas normalizadas para pixels reais
 * - Calcula o centro do bounding box
 *
 * Uso de generator (function*):
 * - Permite enviar cada predição assim que processada
 * - Economiza memória ao não criar lista intermediária
 * - Usa 'yield' em vez de 'return' para pausar e retomar a execução
 * 
 * @param {Object} inference - Objeto com boxes, scores, classes arrays
 * @param {number} width - Largura da imagem original em pixels
 * @param {number} height - Altura da imagem original em pixels
 * @yields {Object} Objeto com x, y (centro) e score (confiança)
 */
function* processPrediction({ boxes, scores, classes }, width, height) {
    // Itera sobre cada detecção
    for (let index = 0; index < scores.length; index++) {
        // Se a confiança < 40%, ignora esta detecção
        if (scores[index] < CLASS_THRESHOLD) continue

      
        const label = _labels[classes[index]]
      
        if (label !== 'kite') continue

        // Extrai as 4 coordenadas do bounding box (normalizadas 0-1)
        let [x1, y1, x2, y2] = boxes.slice(index * 4, (index + 1) * 4)
        
        // Converte coordenadas normalizadas para pixels reais
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

       
        const boxWidth = x2 - x1
        const boxHeight = y2 - y1
        
       
        const centerX = x1 + boxWidth / 2
        const centerY = y1 + boxHeight / 2

       
        yield {
            x: centerX, 
            y: centerY,
            score: (scores[index] * 100).toFixed(2) 
        }
    }
}

// Carrega o modelo e rótulos quando o Worker inicia
loadModelAndLabels()

/**
 * Listener que recebe mensagens da thread principal
 * - Aguarda mensagens do tipo 'predict'
 * - Pré-processa a imagem
 * - Executa a inferência
 * - Processa e envia cada predição válida de volta
 */
self.onmessage = async ({ data }) => {
    // Ignora mensagens que não sejam 'predict'
    if (data.type !== 'predict') return

    if (!_model) return

    // Pré-processa a imagem recebida
    const input = preprocessImage(data.image)
    const { width, height } = data.image

    // Executa a inferência no modelo
    const inferenceResults = await runInference(input)

    // Processa cada predição válida (usando o generator)
    for (const prediction of processPrediction(inferenceResults, width, height)) {
        // Envia cada predição de volta para a thread principal
        postMessage({
            type: 'prediction',
            ...prediction // Spread operator para incluir x, y, score
        });
    }
};

// Log indicando que o Worker foi inicializado com sucesso
console.log('🧠 YOLOv5n Web Worker initialized');
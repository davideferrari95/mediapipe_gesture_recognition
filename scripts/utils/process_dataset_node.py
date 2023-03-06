#!/usr/bin/env python3

import os, cv2
import rospy, rospkg 
import numpy as np
from termcolor import colored

import mediapipe
from mediapipe_gesture_recognition.msg import Keypoint, Hand, Pose, Face

class MediapipeDatasetProcess:

  # Constants
  RIGHT_HAND, LEFT_HAND = True, False

  # Define Hand Landmark Names
  hand_landmarks_names = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP',
                          'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
                          'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP',
                          'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP',
                          'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

  # Define Pose Landmark Names
  pose_landmarks_names = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
                          'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
                          'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST',
                          'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
                          'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                          'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

  def __init__(self):

    # ROS Initialization
    rospy.init_node('mediapipe_dataset_processor_node', anonymous=True)

    # Get Package Path - Get Dataset Folder
    self.package_path = rospkg.RosPack().get_path('mediapipe_gesture_recognition')
    self.DATASET_PATH = os.path.join(self.package_path, r'dataset/Jester Dataset/Videos')

    # Read Mediapipe Modules Parameters
    self.enable_right_hand = rospy.get_param('enable_right_hand', False)
    self.enable_left_hand  = rospy.get_param('enable_left_hand', False)
    self.enable_pose       = rospy.get_param('enable_pose', False)
    self.enable_face       = rospy.get_param('enable_face', False)

    # Select Gesture File
    self.gesture_enabled_folder = ''
    if self.enable_right_hand: self.gesture_enabled_folder += 'Right'
    if self.enable_left_hand:  self.gesture_enabled_folder += 'Left'
    if self.enable_pose:       self.gesture_enabled_folder += 'Pose'
    if self.enable_face:       self.gesture_enabled_folder += 'Face'

    # Debug Print
    print(colored(f'\nFunctions Enabled:\n', 'yellow'))
    print(colored(f'  Right Hand: {self.enable_right_hand}',  'green' if self.enable_right_hand else 'red'))
    print(colored(f'  Left  Hand: {self.enable_left_hand}\n', 'green' if self.enable_left_hand  else 'red'))
    print(colored(f'  Skeleton:   {self.enable_pose}',        'green' if self.enable_pose else 'red'))
    print(colored(f'  Face Mesh:  {self.enable_face}\n',      'green' if self.enable_face else 'red'))

    # Initialize Mediapipe:
    self.mp_drawing = mediapipe.solutions.drawing_utils
    self.mp_drawing_styles = mediapipe.solutions.drawing_styles
    self.mp_holistic = mediapipe.solutions.holistic

    # Initialize Mediapipe Holistic
    self.holistic = self.mp_holistic.Holistic(refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # TODO: Initialise the Walk
    self.gesture_name = ''
    self.video_number = ''

  def newKeypoint(self, landmark, number, name):

    ''' New Keypoint Creation Utility Function '''

    # Assign Keypoint Coordinates
    new_keypoint = Keypoint()
    new_keypoint.x = landmark.x
    new_keypoint.y = landmark.y
    new_keypoint.z = landmark.z
    new_keypoint.v = landmark.visibility

    # Assign Keypoint Number and Name
    new_keypoint.keypoint_number = number
    new_keypoint.keypoint_name = name

    return new_keypoint

  def processHand(self, RightLeft, handResults, image):

    ''' Process Hand Keypoints '''

    # Drawing the Hand Landmarks
    self.mp_drawing.draw_landmarks(
      image,
      handResults.right_hand_landmarks if RightLeft else handResults.left_hand_landmarks,
      self.mp_holistic.HAND_CONNECTIONS,
      self.mp_drawing_styles.get_default_hand_landmarks_style(),
      self.mp_drawing_styles.get_default_hand_connections_style())

    # Create Hand Message
    hand_msg = Hand()
    hand_msg.header.stamp = rospy.Time.now()
    hand_msg.header.frame_id = 'Hand Right Message' if RightLeft else 'Hand Left Message'
    hand_msg.right_or_left = hand_msg.RIGHT if RightLeft else hand_msg.LEFT

    if (((RightLeft == self.RIGHT_HAND) and (handResults.right_hand_landmarks))
     or ((RightLeft == self.LEFT_HAND)  and (handResults.left_hand_landmarks))):

      # Add Keypoints to Hand Message
      for i in range(len(handResults.right_hand_landmarks.landmark if RightLeft else handResults.left_hand_landmarks.landmark)):

        # Append Keypoint
        hand_msg.keypoints.append(self.newKeypoint(handResults.right_hand_landmarks.landmark[i] if RightLeft else handResults.left_hand_landmarks.landmark[i],
                                                   i, self.hand_landmarks_names[i]))

      # Return Hand Keypoint Message
      return hand_msg

  def processPose(self, poseResults, image):

    ''' Process Pose Keypoints '''

    # Drawing the Pose Landmarks
    self.mp_drawing.draw_landmarks(
      image,
      poseResults.pose_landmarks,
      self.mp_holistic.POSE_CONNECTIONS,
      landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

    # Create Pose Message
    pose_msg = Pose()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = 'Pose Message'

    if poseResults.pose_landmarks:

      # Add Keypoints to Pose Message
      for i in range(len(poseResults.pose_landmarks.landmark)):

        # Append Keypoint
        pose_msg.keypoints.append(self.newKeypoint(poseResults.pose_landmarks.landmark[i], i, self.pose_landmarks_names[i]))

      # Return Pose Keypoint Message
      return pose_msg

  def processFace(self, faceResults, image):

    ''' Process Face Keypoints '''

    # Drawing the Face Landmarks
    self.mp_drawing.draw_landmarks(
      image,
      faceResults.face_landmarks,
      self.mp_holistic.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())

    # Create Face Message
    face_msg = Face()
    face_msg.header.stamp = rospy.Time.now()
    face_msg.header.frame_id = 'Face Message'

    if faceResults.face_landmarks:

      # Add Keypoints to Face Message
      for i in range(len(faceResults.face_landmarks.landmark)):

        # Assign Keypoint Coordinates
        new_keypoint = Keypoint()
        new_keypoint.x = faceResults.face_landmarks.landmark[i].x
        new_keypoint.y = faceResults.face_landmarks.landmark[i].y
        new_keypoint.z = faceResults.face_landmarks.landmark[i].z

        # Assign Keypoint Number
        new_keypoint.keypoint_number = i

        # Assign Keypoint Name (468 Landmarks -> Names = FACE_KEYPOINT_1 ...)
        new_keypoint.keypoint_name = f'FACE_KEYPOINT_{i+1}'

        # Append Keypoint
        face_msg.keypoints.append(new_keypoint)

      # Return Face Message
      return face_msg

  def flattenKeypoints(self, pose_msg, left_msg, right_msg, face_msg):

    '''
    Flatten Incoming Messages or Create zeros Vector \n
    Concatenate each Output
    '''

    # Check if Messages are Available and Create Zeros Vectors if Not
    pose    = np.zeros(33 * 4)  if pose_msg  is None else np.array([[res.x, res.y, res.z, res.v] for res in pose_msg.keypoints]).flatten()
    left_h  = np.zeros(21 * 4)  if left_msg  is None else np.array([[res.x, res.y, res.z, res.v] for res in left_msg.keypoints]).flatten()
    right_h = np.zeros(21 * 4)  if right_msg is None else np.array([[res.x, res.y, res.z, res.v] for res in right_msg.keypoints]).flatten()
    face    = np.zeros(468 * 3) if face_msg  is None else np.array([[res.x, res.y, res.z, res.v] for res in face_msg.keypoints]).flatten()

    # Concatenate Data
    return np.concatenate([right_h, left_h, pose, face])

  def processResults(self, image):

    ''' Process the Image to Obtain a Flattened Keypoint Sequence of the Frame '''

    # Instance the ROS Hand, Pose, Face Messages
    left_hand, right_hand, pose, face = Hand(), Hand(), Pose(), Face()

    # Process Left Hand Landmarks
    if self.enable_left_hand: left_hand = self.processHand(self.LEFT_HAND,  self.holistic_results, image)

    # Process Right Hand Landmarks
    if self.enable_right_hand: right_hand = self.processHand(self.RIGHT_HAND, self.holistic_results, image)

    # Process Pose Landmarks
    if self.enable_pose: pose = self.processPose(self.holistic_results, image)

    # Process Face Landmarks
    if self.enable_face: face = self.processFace(self.holistic_results, image)

    # Flatten All the Keypoints
    sequence = self.flattenKeypoints(pose, left_hand, right_hand, face)

    # Return the Flattened Keypoints Sequence
    return sequence

  def saveFrame(self, gesture, video_number, keypoints_sequence):

    ''' Data Save Functions '''

    self.framenumber = 0 

    '''
    Loop to Save the Landmarks Coordinates for Each Frame of Each Video
    The Loop Continue Until the Service Response Keep Itself True with a 30FPS
    '''
    
    # TODO: save all frames of a video in a single file (keypoint_sequence is a vector of vector)

    # Create a Gesture Folder 
    os.makedirs(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_enabled_folder}/', gesture), exist_ok=True)

    # Create a Number Folder for Each Video of the Current Gesture 
    os.makedirs(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_enabled_folder}/', gesture, str(video_number)), exist_ok=True)

    # Export Keypoints Values in the Correct Folder
    npy_path = os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_enabled_folder}/', gesture, str(video_number), str(self.framenumber))

    # Check if This Frame Number Exists and Iterate the Frame Numbers Until the Right FrameNumber
    while os.path.exists(npy_path + '.npy'): 
      self.framenumber = int(self.framenumber) + 1
      npy_path = os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_enabled_folder}/', gesture, str(video_number), str(self.framenumber))

    # Save the Keypoints in the Correct Folder
    np.save(npy_path, keypoints_sequence)

    # Print the Current Gesture and the Video Number
    print(f'Collecting Frames for {gesture} | Video Number: {video_number}  | Frame number:{self.framenumber}')

  def npyfileFiller(self, gesture, video_number):

    # Check if the Video Number is Not Empty
    if not video_number == '':

      # Get the Number of `.npy` Files in the Current Gesture Folder
      npy_path = os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_enabled_folder}/', gesture, str(video_number)) 
      npyfilenumber = os.listdir(npy_path)
      npyfilenumber = len(npyfilenumber)

      # Check if the `.npy` File Number is < 40
      if npyfilenumber < 40:

        # Load the Last `.npy` File
        data = np.load(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_enabled_folder}/', gesture, str(video_number), str(npyfilenumber -1)+".npy"))

        # Copy the Last `.npy` File to Obtain 40 `.npy` Files to Train the NN
        for i in range (40 - npyfilenumber):
          np.save(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_enabled_folder}/', gesture, str(video_number), str(npyfilenumber + i)), data)

  def processDataset(self):

    print('Starting Collection')
    
    # for folder in os.listdir(self.DATASET_PATH):
    #   print(folder)
    #   for video in os.listdir(os.path.join(self.DATASET_PATH, folder)):
    #         print(video)
    
    ''' 
    
    Per salvare il punto di avanzamento durante l'elaborazione del dataset e ripartire da quel punto in caso di interruzione del codice, è possibile utilizzare una combinazione di salvataggio su file e gestione delle eccezioni.

    Ecco un possibile approccio:

        Inizializza una variabile globale, ad esempio last_processed_video, per memorizzare l'ultimo video elaborato.
        Crea una funzione per elaborare un singolo video. All'interno della funzione, inserisci il codice che elabora il video e alla fine aggiorna la variabile last_processed_video con il nome del video elaborato.
        Inserisci un blocco try-except per gestire le eccezioni. All'interno del blocco try, esegui un ciclo che attraversa tutti i video da elaborare. Per ogni video, verifica se il suo nome è maggiore del nome dell'ultimo video elaborato (puoi usare la funzione sorted per ordinare i nomi dei file in ordine alfabetico). Se il video non è stato ancora elaborato, esegui la funzione che elabora il video. All'interno del blocco except, salva la variabile last_processed_video su un file di testo.
        Alla successiva esecuzione del codice, controlla se esiste un file di testo che contiene l'ultimo video elaborato e carica la variabile last_processed_video dal file. In questo modo, sarai in grado di riprendere l'elaborazione dal punto in cui l'hai interrotta.

    Ecco un esempio di codice Python che implementa questo approccio:

    import os

    # Inizializza la variabile last_processed_video
    last_processed_video = ""

    # Definisci la funzione per elaborare un singolo video
    def process_video(video_path):
        # Inserisci qui il codice per elaborare il video
        # ...
        # Aggiorna la variabile last_processed_video con il nome del video elaborato
        global last_processed_video
        last_processed_video = os.path.basename(video_path)

    # Esegui l'elaborazione dei video
    try:
        # Carica la lista dei video da elaborare
        video_list = sorted(os.listdir("path/to/videos"))

        # Cerca l'ultimo video elaborato
        if os.path.exists("last_processed_video.txt"):
            with open("last_processed_video.txt", "r") as f:
                last_processed_video = f.read().strip()

        # Elabora i video
        for video_name in video_list:
            video_path = os.path.join("path/to/videos", video_name)
            if video_name > last_processed_video:
                process_video(video_path)

    except KeyboardInterrupt:
        # In caso di interruzione con CTRL+C, salva l'ultimo video elaborato
        with open("last_processed_video.txt", "w") as f:
            f.write(last_processed_video)

    Nel codice sopra, la funzione os.path.basename viene utilizzata per ottenere il nome del file dal percorso completo del video. La funzione os.path.exists viene utilizzata per controllare se il file "last_processed_video.txt" esiste già. La funzione strip viene utilizzata per rimuovere eventuali spazi bianchi dal contenuto del file. Infine, la variabile KeyboardInterrupt viene utilizzata per gestire l'interruzione del cod

    '''
    
    ''' 
    possiamo fare un txt che, dopo aver salvato un video, viene aggiornato con nome_gesto + numero_video
    con dei for annidati processiamo tutte le cartelle e tutti i video al loro interno (in ordine alfabetico!)
    controlliamo prima la cartella del gesto, se siamo al gesto "No gesture" saltiamo "Doing other things" e "Drumming Fingers" in quando precedenti in ordine alfabetico
    dentro la cartella se abbiamo salvato il video 100, partiamo dal 101, ignorando tutti quelli prima
    '''
    
    ''' 
    DA CAMBIARE:
    
      1. Salvare ogni video con un singolo file (.pkl, .npy, altri) organizzati in modo decente; non possiamo avere 148000 * 40+ file .npy, già col dataset la cartella è pesantissima, figuriamoci con tutti quei files extra...
         Anzi, forse salverei un file per ogni gesto addirittura, utilizzando pickle o simili e salvando una custom dataclass fatta da un array contenente n_video array di frame
         Anche perchè in futuro se vorremo fare un training con solo la face o con tutto ecc non possiamo avere 148000 * 40+ * 3/4/5 volte...
      
      2. La funzione npyfiller secondo me messa qua è sbagliato. Riflettendoci questo è un nodo che processa i video, aggiungere dei dati exta è come se andassimo a "modificare" i video
         La cosa più corretta secondo me è salvare qua i video come sono, poi nella NN usare una tecnica di filling come la "zero-padding" descritta sotto.
         
      3. Inserire assolutamente una funzione che gestica l'interruzione con CRTL+C, deve funzionare. Col txt come descritto sopra forse potrebbe funzionare.
      
    agiustate queste cose direi che possiamo far partire il processo dei video 
    '''

    # Read Every File in the Directory
    for root, dirs, files in sorted(os.walk(self.DATASET_PATH)):

      # Get the Current Subfolder
      current_subdir = os.path.basename(root)

      # Read Every Video in Every Subfolder
      for filename in files:

        # FIX: Fill the Frames Gap -> Perchè è qui ? Gesture Name è letto dopo
        self.npyfileFiller(self.gesture_name, self.video_number)

        # Take the Gesture Name from the Current Folder
        self.gesture_name = os.path.splitext(current_subdir)[0]

        # Take the Video Number
        self.video_number = os.path.splitext(filename)[0]

        # Print the Current Observed Video
        #print("\nCurrent gesture:", gesture_name,"Current video:", video_number)

        # Check if the File is a Video
        if not filename.endswith(('.mp4', '.avi', '.mov')):
          continue

        # Get the Full Path of the Video for Each Gesture
        video_path = os.path.join(root, filename)

        # Open the Video
        video_cap = cv2.VideoCapture(video_path)

        while video_cap.isOpened() and not rospy.is_shutdown():

          # Read Webcam Im
          success, image = video_cap.read()

          if not success:

            print('Video Finished\n')
            break

          # To Improve Performance -> Process the Image as Not-Writeable
          image.flags.writeable = False
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          # Get Holistic Results from Mediapipe Holistic
          self.holistic_results = self.holistic.process(image)

          # To Draw the Annotations -> Set the Image Writable
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          # Process Mediapipe Results
          sequence = self.processResults(image)

          # Show and Flip the Image Horizontally for a Selfie-View Display.
          cv2.imshow('MediaPipe Landmarks', cv2.flip(image, 1))
          if cv2.waitKey(5) & 0xFF == 27:
            break

          # TODO: move save data out of the while, appending frame in a vector
          # Save Frame Data
          self.saveFrame(self.gesture_name, self.video_number, sequence)

        # Close Video Cap
        video_cap.release()

    # Remove this function, the differences have to be fixed in the training node 
    self.npyfileFiller(self.gesture_name, self.video_number)
    
    ''' 
    Sì, posso aiutarti a creare una rete neurale per il riconoscimento dei gesti in 3D. Esistono diverse architetture di reti neurali che possono essere utilizzate per questa attività, ma una delle più comuni è la CNN (Convolutional Neural Network). Di seguito è riportato un esempio di architettura di rete neurale CNN per il riconoscimento dei gesti in 3D:

    Input Layer: L'input layer della rete neurale accetta i dati di input che rappresentano il gesto in 3D. Ad esempio, i dati di input potrebbero essere una serie di frame 3D che rappresentano un movimento della mano o del corpo.
    Convolutional Layer: Il primo livello della rete neurale è il livello convoluzionale, che estrae le feature dei dati di input. In particolare, il livello convoluzionale applica una serie di filtri convoluzionali sui dati di input per creare una mappa delle feature. Ad esempio, i filtri convoluzionali possono essere utilizzati per rilevare i bordi o le forme dei gesti.
    Pooling Layer: Il livello di pooling riduce la dimensione della mappa delle feature creata dal livello convoluzionale. Ciò è utile perché riduce il numero di parametri della rete neurale e, quindi, rende la rete neurale più veloce e meno suscettibile all'overfitting.
    Convolutional Layer: Un altro livello convoluzionale può essere utilizzato per estrarre ulteriori feature dal pool delle feature creato dal primo livello convoluzionale.
    Pooling Layer: Un altro livello di pooling può essere utilizzato per ridurre ulteriormente la dimensione della mappa delle feature.
    Fully Connected Layer: Il layer fully connected accetta l'output dell'ultimo livello di pooling e lo trasforma in un vettore di feature unidimensionale. Questo vettore di feature viene quindi passato attraverso una serie di layer fully connected che applicano una funzione di attivazione, come ad esempio la funzione ReLU.
    Output Layer: Il layer di output contiene un numero di neuroni corrispondente al numero di classi di gesti da riconoscere. Ad esempio, se ci sono cinque classi di gesti (ad esempio, pugno, apertura della mano, palmo, dito indice e pollice), il layer di output conterrebbe cinque neuroni. La funzione di attivazione utilizzata in questo livello dipende dal tipo di problema di classificazione. Ad esempio, per un problema di classificazione binaria (due classi), si può utilizzare la funzione di attivazione sigmoid, mentre per un problema di classificazione multiclasse si può utilizzare la funzione di attivazione softmax.

    Training: Dopo aver costruito l'architettura della rete neurale, è necessario allenarla utilizzando un set di dati di training. Durante l'allenamento, la rete neurale cerca di minimizzare la funzione di costo, che misura la differenza tra la predizione della rete neurale e il valore di output corretto.
    Testing: Dopo aver allenato la rete neurale, è

    ---
    
    e se ogni gesto ha un numero diverso di frame ?

    Se ogni gesto ha un numero diverso di frame, è possibile utilizzare una tecnica chiamata "zero-padding" per gestire questa situazione. In pratica, è possibile aggiungere degli zeri al termine dei gesti più brevi, in modo da ottenere un insieme di gesti con la stessa lunghezza. In questo modo, tutti i gesti avranno la stessa dimensione e potranno essere utilizzati come input per la rete neurale.
    Ad esempio, supponiamo che il gesto più breve abbia una lunghezza di 50 frame e il gesto più lungo abbia una lunghezza di 100 frame. In questo caso, è possibile aggiungere 50 zeri al termine dei gesti più brevi in modo da ottenere un insieme di gesti con una lunghezza di 100 frame. In questo modo, tutti i gesti avranno la stessa lunghezza e potranno essere utilizzati come input per la rete neurale.
    Tuttavia, è importante notare che l'aggiunta di zeri può aumentare la dimensione dei dati di input e quindi rendere la rete neurale più lenta o richiedere più memoria. Inoltre, l'aggiunta di zeri potrebbe non essere la soluzione migliore se la differenza di lunghezza tra i gesti è troppo grande. In tal caso, potrebbe essere necessario utilizzare tecniche di pre-elaborazione dei dati, come ad esempio l'interpolazione o il campionamento, per ridurre la differenza di lunghezza tra i gesti.
    '''

    # Print Finish of All Videos
    print('All Video Processed')

if __name__ == '__main__':

  # Create Mediapipe Dataset Process Class
  MediapipeProcess = MediapipeDatasetProcess()

  # Mediapipe Dataset Process Function
  MediapipeProcess.processDataset()

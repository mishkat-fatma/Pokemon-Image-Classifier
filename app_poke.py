import streamlit as st
from PIL import Image
#from Work import classify
import base64
import tensorflow as tf
import numpy as np

# Title and header
st.markdown(
    """
    <div style="text-align: center; color: white; padding: 10px;">
        <h1 style="font-family: 'Comic Sans MS', cursive, sans-serif; color: yellow; text-shadow: 2px 2px #3b4cca;">
            <span style="color: yellow; text-shadow: 2px 2px #3b4cca;">PokéMon</span> Image Classifier
        </h1>
        <h2 style="font-family: 'Verdana', sans-serif; color: yellow; text-shadow: 2px 2px #3b4cca;">
            Snap, Upload, and Identify Your Favorite Pokémon!
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.image("Ash_Ketchum_World_Champion_Screenshot_4.0.jpg")
st.markdown(
    """
    <div style="background-color: white; padding: 20px; border-radius: 10px;">
        <p style="font-size: 18px; color: black;">
            This application allows you to upload an image of a Pokémon, and it will predict which Pokémon it is.
            Simply upload an image using the file uploader below, and our model will handle the rest.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


file = st.file_uploader('Upload your Pokémon image', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')


model=tf.keras.models.load_model("myMLmodel_mobilenet.keras")

class_names = ['Abra',
               'Aerodactyl',
               'Alakazam',
               'Alolan Sandslash',
               'Arbok',
               'Arcanine',
               'Articuno',
               'Beedrill',
               'Bellsprout',
               'Blastoise',
               'Bulbasaur',
               'Butterfree',
               'Caterpie',
               'Chansey',
               'Charizard',
               'Charmander',
               'Charmeleon',
               'Clefable',
               'Clefairy',
               'Cloyster',
               'Cubone',
               'Dewgong',
               'Diglett',
               'Ditto',
               'Dodrio',
               'Doduo',
               'Dragonair',
               'Dragonite',
               'Dratini',
               'Drowzee',
               'Dugtrio',
               'Eevee',
               'Ekans',
               'Electabuzz',
               'Electrode',
               'Exeggcute',
               'Exeggutor',
               'Farfetchd',
               'Fearow',
               'Flareon',
               'Gastly',
               'Gengar',
               'Geodude',
               'Gloom',
               'Golbat',
               'Goldeen',
               'Golduck',
               'Golem',
               'Graveler',
               'Grimer',
               'Growlithe',
               'Gyarados',
               'Haunter',
               'Hitmonchan',
               'Hitmonlee',
               'Horsea',
               'Hypno',
               'Ivysaur',
               'Jigglypuff',
               'Jolteon',
               'Jynx',
               'Kabuto',
               'Kabutops',
               'Kadabra',
               'Kakuna',
               'Kangaskhan',
               'Kingler',
               'Koffing',
               'Krabby',
               'Lapras',
               'Lickitung',
               'Machamp',
               'Machoke',
               'Machop',
               'Magikarp',
               'Magmar',
               'Magnemite',
               'Magneton',
               'Mankey',
               'Marowak',
               'Meowth',
               'Metapod',
               'Mew',
               'Mewtwo',
               'Moltres',
               'MrMime',
               'Muk',
               'Nidoking',
               'Nidoqueen',
               'Nidorina',
               'Nidorino',
               'Ninetales',
               'Oddish',
               'Omanyte',
               'Omastar',
               'Onix',
               'Paras',
               'Parasect',
               'Persian',
               'Pidgeot',
               'Pidgeotto',
               'Pidgey',
               'Pikachu',
               'Pinsir',
               'Poliwag',
               'Poliwhirl',
               'Poliwrath',
               'Ponyta',
               'Porygon',
               'Primeape',
               'Psyduck',
               'Raichu',
               'Rapidash',
               'Raticate',
               'Rattata',
               'Rhydon',
               'Rhyhorn',
               'Sandshrew',
               'Sandslash',
               'Scyther',
               'Seadra',
               'Seaking',
               'Seel',
               'Shellder',
               'Slowbro',
               'Slowpoke',
               'Snorlax',
               'Spearow',
               'Squirtle',
               'Starmie',
               'Staryu',
               'Tangela',
               'Tauros',
               'Tentacool',
               'Tentacruel',
               'Vaporeon',
               'Venomoth',
               'Venonat',
               'Venusaur',
               'Victreebel',
               'Vileplume',
               'Voltorb',
               'Vulpix',
               'Wartortle',
               'Weedle',
               'Weepinbell',
               'Weezing',
               'Wigglytuff',
               'Zapdos',
               'Zubat']

pokemon_descriptions=[
    "Abra: A psychic-type Pokémon known for its teleportation abilities.",
    "Aerodactyl: A prehistoric flying Pokémon with sharp fangs and a fierce demeanor.",
    "Alakazam: Evolved form of Abra, possessing extraordinary intelligence and psychic powers.",
    "Alolan Sandslash: A regional variant of Sandslash adapted to icy environments, sporting icy spines.",
    "Arbok: A serpent-like Pokémon with a hood patterned with menacing designs.",
    "Arcanine: A majestic and loyal fire-type Pokémon resembling a legendary creature from Eastern mythology.",
    "Articuno: A legendary ice and flying-type Pokémon known for its grace and beauty.",
    "Beedrill: A large bee-like Pokémon armed with a venomous stinger.",
    "Bellsprout: A plant-type Pokémon with a bell-shaped head and vine-like appendages.",
    "Blastoise: The final evolved form of Squirtle, a powerful water-type Pokémon with cannons on its back.",
    "Bulbasaur: A grass and poison-type Pokémon with a plant bulb on its back.",
    "Butterfree: An elegant butterfly Pokémon known for its powdery scales.",
    "Caterpie: A small bug-type Pokémon with a voracious appetite for leaves.",
    "Chansey: A kind-hearted Pokémon that carries eggs to aid injured Pokémon and Trainers.",
    "Charizard: The final evolved form of Charmander, a fire and flying-type Pokémon with dragon-like features.",
    "Charmander: A fire-type Pokémon with a flame at the tip of its tail.",
    "Charmeleon: The evolved form of Charmander, a fire-type Pokémon with a fiery temper.",
    "Clefable: A fairy-type Pokémon known for its whimsical appearance and magical abilities.",
    "Clefairy: A cute fairy-type Pokémon that is often found dancing under the moonlight.",
    "Cloyster: A water and ice-type Pokémon protected by a hard shell, resembling a bivalve.",
    "Cubone: A ground-type Pokémon that wears the skull of its deceased mother as a helmet.",
    "Dewgong: A water and ice-type Pokémon resembling a seal with a horn.",
    "Diglett: A ground-type Pokémon that lives underground and pops up from the earth.",
    "Ditto: A unique Pokémon that can transform into any other Pokémon it sees.",
    "Dodrio: A three-headed bird Pokémon known for its speed and agility.",
    "Doduo: A two-headed bird Pokémon that runs at high speeds on its powerful legs.",
    "Dragonair: The elegant evolved form of Dratini, a dragon-type Pokémon with serpentine features.",
    "Dragonite: The final evolved form of Dratini, a dragon and flying-type Pokémon with immense strength.",
    "Dratini: A dragon-type Pokémon with a long, snake-like body.",
    "Drowzee: A psychic-type Pokémon known for its ability to induce sleep with its pendulum.",
    "Dugtrio: A trio of ground-type Pokémon that share a single body, each with its own head.",
    "Eevee: A unique Pokémon with the ability to evolve into various forms based on its environment and conditions.",
    "Ekans: A snake-like Pokémon with a venomous bite and the ability to constrict its prey.",
    "Electabuzz: An electric-type Pokémon capable of generating powerful electrical currents.",
    "Electrode: A round, explosive Pokémon that rolls around at high speeds.",
    "Exeggcute: A group of small egg-like Pokémon that communicate through telepathy.",
    "Exeggutor: A grass and psychic-type Pokémon with multiple heads and a tall, coconut tree-like body.",
    "Farfetch'd: A bird-like Pokémon that carries a leek stalk as its weapon of choice.",
    "Fearow: A large bird Pokémon known for its sharp beak and keen eyesight.",
    "Flareon: A fire-type Pokémon known for its fluffy fur and fiery mane.",
    "Gastly: A ghost and poison-type Pokémon composed of toxic gases.",
    "Gengar: A mischievous ghost-type Pokémon that delights in causing fear and mischief.",
    "Geodude: A rock and ground-type Pokémon with rocky, humanoid features.",
    "Gloom: A grass and poison-type Pokémon known for its unpleasant aroma.",
    "Golbat: A bat-like Pokémon with large wings and sharp fangs.",
    "Goldeen: A fish-like Pokémon with a golden body and a horn on its forehead.",
    "Golduck: A water-type Pokémon known for its psychic powers and sleek appearance.",
    "Golem: The final evolved form of Geodude, a rock and ground-type Pokémon with immense strength.",
    "Graveler: The evolved form of Geodude, a rock and ground-type Pokémon with a rugged appearance.",
    "Grimer: A sludge-like Pokémon composed of toxic substances.",
    "Growlithe: A loyal fire-type Pokémon resembling a small, orange dog with a bushy mane.",
    "Gyarados: A fierce water and flying-type Pokémon known for its destructive power and temper.",
    "Haunter: The evolved form of Gastly, a ghost and poison-type Pokémon that haunts abandoned buildings.",
    "Hitmonchan: A fighting-type Pokémon known for its boxing skills and powerful punches.",
    "Hitmonlee: A fighting-type Pokémon that specializes in powerful kicks and agile movements.",
    "Horsea: A small water-type Pokémon resembling a seahorse.",
    "Hypno: A psychic-type Pokémon known for its ability to hypnotize opponents with its pendulum.",
    "Ivysaur: The evolved form of Bulbasaur, a grass and poison-type Pokémon with a budding flower on its back.",
    "Jigglypuff: A fairy-type Pokémon known for its soothing voice and ability to put others to sleep with its song.",
    "Jolteon: An electric-type Pokémon with sharp, electrified fur and lightning-fast reflexes.",
    "Jynx: A humanoid ice and psychic-type Pokémon known for its graceful movements and hypnotic dance.",
    "Kabuto: A prehistoric water and rock-type Pokémon resembling a trilobite.",
    "Kabutops: The final evolved form of Kabuto, a fearsome water and rock-type Pokémon with scythe-like arms.",
    "Kadabra: The evolved form of Abra, a psychic-type Pokémon with highly developed mental abilities.",
    "Kakuna: The cocoon stage in the evolution of Weedle, a bug-type Pokémon known for its hardened shell.",
    "Kangaskhan: A maternal Pokémon known for its protective nature towards its young.",
    "Kingler: A large crab-like Pokémon with powerful pincers.",
    "Koffing: A toxic gas Pokémon known for emitting foul-smelling gases.",
    "Krabby: A small crab-like Pokémon with a pincer on each arm.",
     "Lapras: A gentle water and ice-type Pokémon resembling a large sea creature with a musical cry.",
    "Lickitung: A pink, tongue-like Pokémon with a long, stretchable tongue used to ensnare prey.",
    "Machamp: The final evolved form of Machop, a powerful fighting-type Pokémon with four muscular arms.",
    "Machoke: The evolved form of Machop, a muscular fighting-type Pokémon known for its strength.",
    "Machop: A fighting-type Pokémon known for its impressive physical prowess and determination.",
    "Magikarp: A weak and helpless fish Pokémon that evolves into the powerful Gyarados.",
    "Magmar: A fire-type Pokémon known for its fiery personality and intense heat.",
    "Magnemite: A steel and electric-type Pokémon with a magnet for a body that attracts metal objects.",
    "Magneton: The evolved form of Magnemite, a cluster of three Magnemite linked by strong magnetic forces.",
    "Mankey: A small, aggressive fighting-type Pokémon known for its short temper.",
    "Marowak: The evolved form of Cubone, a ground-type Pokémon that wields a bone as a weapon.",
    "Meowth: A cat-like Pokémon known for its fondness for shiny objects and ability to learn Pay Day.",
    "Metapod: The cocoon stage in the evolution of Caterpie, a bug-type Pokémon with a tough outer shell.",
    "Mew: A mythical psychic-type Pokémon said to possess the genetic composition of all Pokémon.",
    "Mewtwo: A powerful psychic-type Pokémon created through genetic manipulation, possessing immense psychic abilities.",
    "Moltres: A legendary fire and flying-type Pokémon representing the element of fire.",
    "Mr. Mime: A humanoid psychic-type Pokémon known for its pantomime abilities and invisible barriers.",
    "Muk: The evolved form of Grimer, a toxic sludge Pokémon known for its foul odor and corrosive substances.",
    "Nidoking: The final evolved form of Nidoran♂, a powerful poison and ground-type Pokémon with a sturdy build.",
    "Nidoqueen: The final evolved form of Nidoran♀, a strong poison and ground-type Pokémon known for its protective nature.",
    "Nidorina: The evolved form of Nidoran♀, a poison-type Pokémon with a gentle demeanor.",
    "Nidorino: The evolved form of Nidoran♂, a poison-type Pokémon with a fierce temperament.",
    "Ninetales: The final evolved form of Vulpix, a majestic fire-type Pokémon with nine tails.",
    "Oddish: A small plant-type Pokémon resembling a blue plant bulb with leaves on its head.",
    "Omanyte: A prehistoric water and rock-type Pokémon resembling a snail or ammonite.",
    "Omastar: The final evolved form of Omanyte, a powerful water and rock-type Pokémon with tentacle-like arms.",
    "Onix: A rock and ground-type Pokémon known for its massive size and burrowing abilities.",
    "Paras: A bug and grass-type Pokémon resembling a mushroom with insectoid features.",
    "Parasect: The evolved form of Paras, a bug and grass-type Pokémon controlled by the parasitic mushroom on its back.",
    "Persian: The evolved form of Meowth, a sleek and elegant cat-like Pokémon known for its speed.",
    "Pidgeot: The final evolved form of Pidgey, a majestic bird Pokémon with powerful wings.",
    "Pidgeotto: The evolved form of Pidgey, a bird Pokémon with keen eyesight and sharp talons.",
    "Pidgey: A small bird Pokémon known for its gentle nature and ability to fly short distances.",
    "Pikachu: An iconic electric-type Pokémon known for its lightning bolt-shaped tail and cute appearance.",
    "Pinsir: A bug-type Pokémon known for its powerful pincers capable of crushing hard objects.",
    "Poliwag: A water-type Pokémon with a swirl pattern on its belly and a tadpole-like appearance.",
    "Poliwhirl: The evolved form of Poliwag, a water-type Pokémon with spiral patterns on its belly.",
    "Poliwrath: The evolved form of Poliwhirl, a powerful water and fighting-type Pokémon known for its martial arts skills.",
    "Ponyta: A fire-type Pokémon resembling a small, fiery horse with a flaming mane.",
    "Porygon: A virtual Pokémon created through computer programming, capable of moving freely in cyberspace.",
    "Primeape: The evolved form of Mankey, a fierce and aggressive fighting-type Pokémon.",
    "Psyduck: A water-type Pokémon known for its constant headaches and psychic abilities.",
    "Raichu: The evolved form of Pikachu, an electric-type Pokémon with enhanced electrical powers.",
    "Rapidash: The evolved form of Ponyta, a majestic fire-type Pokémon known for its incredible speed.",
    "Raticate: The evolved form of Rattata, a rat-like Pokémon known for its sharp fangs.",
    "Rattata: A small, rodent-like Pokémon known for its quickness and tendency to gnaw on objects.",
    "Rhydon: The final evolved form of Rhyhorn, a massive ground and rock-type Pokémon with a drill-like horn.",
    "Rhyhorn: A ground and rock-type Pokémon with a tough, rocky hide and a horn on its nose.",
    "Sandshrew: A ground-type Pokémon with a tough, sand-colored hide and claws for digging.",
    "Sandslash: The evolved form of Sandshrew, a ground-type Pokémon with sharp claws and a spiky back.",
    "Scyther: A bug and flying-type Pokémon resembling a mantis with scythe-like arms.",
    "Seadra: The evolved form of Horsea, a water-type Pokémon known for its swimming abilities and sharp fins.",
    "Seaking: The evolved form of Goldeen, a water-type Pokémon known for its beautiful fins and graceful movements.",
    "Seel: A water-type Pokémon resembling a seal with a horn on its head.",
    "Shellder: A water-type Pokémon resembling a clam with a blue shell and a tongue-like appendage.",
    "Slowbro: The evolved form of Slowpoke, a water and psychic-type Pokémon with a Shellder attached to its tail.",
    "Slowpoke: A slow-moving water and psychic-type Pokémon known for its absent-minded demeanor.",
    "Snorlax: A giant, sleeping Pokémon known for its insatiable appetite and incredible strength.",
    "Spearow: A small bird Pokémon known for its sharp beak and territorial behavior.",
    "Squirtle: A water-type Pokémon with a tough shell and the ability to shoot water from its mouth.",
    "Starmie: The evolved form of Staryu, a water and psychic-type Pokémon resembling a starfish with a gemstone core.",
    "Staryu: A water-type Pokémon resembling a starfish with a red jewel-like core.",
    "Tangela: A grass-type Pokémon covered in tangled vines that it uses to ensnare its opponents.",
    "Tauros: A wild bull-like Pokémon known for its aggressive nature and charging attacks.",
    "Tentacool: A water and poison-type Pokémon with tentacles that exude toxic substances.",
    "Tentacruel: The evolved form of Tentacool, a larger and more powerful water and poison-type Pokémon.",
    "Vaporeon: A water-type Pokémon known for its ability to control water molecules and dissolve into water.",
    "Venomoth: A bug and poison-type Pokémon with powdery wings that scatter toxic scales.",
    "Venonat: A bug and poison-type Pokémon resembling a fuzzy insect with large eyes.",
    "Venusaur: The final evolved form of Bulbasaur, a powerful grass and poison-type Pokémon with a large flower on its back.",
    "Victreebel: The evolved form of Weepinbell, a grass and poison-type Pokémon with a large, carnivorous plant trap.",
    "Vileplume: The evolved form of Gloom, a grass and poison-type Pokémon known for its toxic spores.",
    "Voltorb: An electric-type Pokémon resembling a Poké Ball with a face, known for its explosive temperament.",
    "Vulpix: A fox-like fire-type Pokémon with six orange tails, each emitting a fiery aura.",
    "Wartortle: The evolved form of Squirtle, a water-type Pokémon with a tough shell and water cannons on its back.",
    "Weedle: A small bug and poison-type Pokémon with a sharp stinger on its head.",
    "Weepinbell: The evolved form of Bellsprout, a plant-type Pokémon with a large, bell-shaped body.",
    "Weezing: The evolved form of Koffing, a toxic gas Pokémon known for its dual exhaust pipes.",
    "Wigglytuff: The evolved form of Jigglypuff, a fairy-type Pokémon with a large, round body and a soothing voice.",
    "Zapdos: A legendary electric and flying-type Pokémon known as the 'Electric Pokémon' and one of the legendary birds of Kanto.",
    "Zubat: A bat-like poison and flying-type Pokémon known for its echolocation abilities and nocturnal habits."]

def classify(image, model, class_names, pokemon_descriptions):
    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)



    predictions = model.predict(image_array)


    max_index = np.argmax(predictions[0])
    class_name = class_names[max_index]
    confidence_score = round(predictions[0][max_index] * 100, 2)
    description = pokemon_descriptions[max_index]

    return class_name, confidence_score, description
if file is not None:
    image=Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)

    classn, confidence_score,description = classify(image, model, class_names,pokemon_descriptions)
    st.write("Image Classified!")
    st.write("## {}".format(classn))
    st.write("### Confidence_Score(%): {}".format(confidence_score))
    st.write("### Description: {}".format(description))

st.markdown(
    """
    ### Model Description


    This is a deep learning model based on computer vision to classify uploaded images into 150 distinct categories. 
    Initially, images are resized to meet the model's input specifications. The model employs transfer learning, utilizing 
    the ResNet101v2 architecture to perform image categorization. At the final stage, a Softmax layer computes the probabilities 
    of the image being one of the 150 Pokémon classes. 

    Feel free to experiment with various Pokémon images!

    """,
    unsafe_allow_html=False
)

st.markdown(
    """
    Developed by Aditya
    """,
    unsafe_allow_html=True
)

st.image("pokemon-party.jpg", use_column_width=True)  # Replace with your image file

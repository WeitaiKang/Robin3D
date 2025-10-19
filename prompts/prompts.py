ID_format = "<OBJ{:03}>"

obj_caption_wid_prompt = [
    "Portray the visual characteristics of the <id>.",
    # "Detail the outward presentation of the <id>.",
    # "Provide a depiction of the <id>'s appearance.",
    # "Illustrate how the <id> looks.",
    # "Describe the visual aspects of the <id>.",
    # "Convey the physical attributes of the <id>.",
    # "Outline the external features of the <id>.",
    # "Render the appearance of the <id> in words.",
    # "Depict the outward form of the <id>.",
    # "Elaborate on the visual representation of the <id>."
]

multi3dref_prompt = [
    "Are there any objects fitting the description of \"<description>\"? If so, kindly provide the IDs for those objects.",
    # "Do any objects match the description of \"<description>\"? If they do, please share the IDs of those objects.",
    # "Is there anything that matches the description \"<description>\"? If yes, please share the IDs of those objects.",
    # "Are there objects that correspond to the description \"<description>\"? If there are, kindly list their IDs.",
    # "Does anything fit the description of \"<description>\"? If it does, could you list the IDs for those objects?",
    # "Are there objects described as \"<description>\"? If there are, please provide the IDs for those objects.",
    # "Have any objects been described as \"<description>\"? If so, please share the IDs of those objects.",
    # "Do any objects meet the criteria of \"<description>\"? If they do, kindly provide the IDs of those objects.",
    # "Are there objects with the attributes of \"<description>\"? If there are, please list their IDs.",
    # "Are there any objects that correspond to the description \"<description>\"? If yes, could you share the IDs for those objects?"
]

region_caption_prompt = [
    "Describe the area surrounding {}.",
    # "Provide a description of the locality around {}.",
    # "Characterize the zone centered on {}.",
    # "Depict the surroundings of {}.",
    # "Illustrate the region with {} at its core.",
    # "Give a portrayal of the area focused around {}.",
    # "Offer a depiction of the vicinity of {}.",
    # "Summarize the setting adjacent to {}.",
    # "Explain the environment encircling {}.",
    # "Detail the sector that encompasses {}."
]

grounding_prompt = [
    "Share the ID of the object that best fits the description \"<description>\".",
    # "Kindly provide the ID of the object that closely matches the description \"<description>\".",
    # "What is the ID of the object that aligns with the description \"<description>\"?",
    # "Identify the ID of the object that closely resembles the description \"<description>\".",
    # "What's the ID of the object that corresponds to the description \"<description>\"?",
    # "Give the ID of the object that most accurately describes the description \"<description>\".",
    # "Share the ID of the object that best corresponds to the description \"<description>\".",
    # "Identify the ID of the object that closely aligns with the description \"<description>\".",
    # "What is the ID of the object that matches the description \"<description>\"?"
]

scanrefer_prompt = [
    "According to the given description, \"<description>\" please provide the ID of the object that has the closest match to this description."
]

groundedscenecap_prompt = [
    "Share all the IDs of the objects that fit the description \"<description>\".",
    # "Kindly provide all the IDs of objects that closely match the description \"<description>\".",
    # "What are the IDs of the objects that align with the description \"<description>\"?",
    # "Identify all the IDs of objects that closely resemble the description \"<description>\".",
    # "What are the IDs of the objects that correspond to the description \"<description>\"?",
    # "Give all the IDs of objects that most accurately describe the description \"<description>\".",
    # "Share all the IDs of objects that best correspond to the description \"<description>\".",
    # "Identify all the IDs of objects that closely align with the description \"<description>\".",
]

partial_grounding_prompt = [
    "If you can, please share the ID of the object that fits the description \"<description>\".",
    # "I remember there is an object that fits the description \"<description>\", but I am not one hundred percent sure. If it does, please share the ID of the object.",
    # "I think, but do not really remember, that there is an object that fits the description \"<description>\". If it does, please share the ID of the object.",
    # "I believe there is an object that matches the description \"<description>\", though I am not entirely certain. If so, please share its ID.",
    # "I vaguely recall an object that matches the description \"<description>\". If it is present, could you provide its ID?",
    # "I am uncertain if there is an object that corresponds to the description \"<description>\". If so, could you share its ID?",
]

scan2cap_prompt = [
    "Begin by detailing the visual aspects of the <id> before delving into its spatial context among other elements within the scene.",
    # "First, depict the physical characteristics of the <id>, followed by its placement and interactions within the surrounding environment.",
    # "Describe the appearance of the <id>, then elaborate on its positioning relative to other objects in the scene.",
    # "Paint a picture of the visual attributes of <id>, then explore how it relates spatially to other elements in the scene.",
    # "Start by articulating the outward features of the <id>, then transition into its spatial alignment within the broader scene.",
    # "Provide a detailed description of the appearance of <id> before analyzing its spatial connections with other elements in the scene.",
    # "Capture the essence of the appearance of <id>, then analyze its spatial relationships within the scene's context.",
    # "Detail the physical characteristics of the <id> and subsequently examine its spatial dynamics amidst other objects in the scene.",
    # "Describe the visual traits of <id> first, then elucidate its spatial arrangements in relation to neighboring elements.",
    # "Begin by outlining the appearance of <id>, then proceed to illustrate its spatial orientation within the scene alongside other objects."
]

pointedcap_prompt = [
    # "If you can, please detail the visual attributes of the <category> I click \"<click>\", then explore how it relates spatially to other elements in the scene. And please provide the ID of the object.",
    "If you can, please detail the visual aspects of the object I click \"<click>\" before delving into its spatial context among other elements within the scene. And please provide the ID of the object.",
    # "First, depict the physical characteristics of the place I click \"<click>\", followed by its placement and interactions within the surrounding environment. And please provide the ID of the object.",
    # "Describe the appearance of the place I click \"<click>\", then elaborate on its positioning relative to other objects in the scene. And please provide the ID of the object.",
    # "Paint a picture of the visual attributes of the place I click \"<click>\", then explore how it relates spatially to other elements in the scene. And please provide the ID of the object.",
    # "Start by articulating the outward features of the place I click \"<click>\", then transition into its spatial alignment within the broader scene. And please provide the ID of the object.",
    # "Provide a detailed description of the appearance of the place I click \"<click>\" before analyzing its spatial connections with other elements in the scene. And please provide the ID of the object.",
    # "Capture the essence of the appearance of the place I click \"<click>\", then analyze its spatial relationships within the scene's context. And please provide the ID of the object.",
    # "Detail the physical characteristics of the place I click \"<click>\" and subsequently examine its spatial dynamics amidst other objects in the scene. And please provide the ID of the object.",
    # "Describe the visual traits of the place I click \"<click>\" first, then elucidate its spatial arrangements in relation to neighboring elements. And please provide the ID of the object.",
    # "Begin by outlining the appearance of the place I click \"<click>\", then proceed to illustrate its spatial orientation within the scene alongside other objects. And please provide the ID of the object."
]

nr3d_caption_prompt = [
    "Detail the spatial positioning of the <id> amidst surrounding elements.",
    # "Illustrate the <id>'s placement relative to its environment.",
    # "Explain the <id>'s location in correlation with nearby items.",
    # "Elaborate on the <id>'s spatial context within the scene.",
    # "Describe how the <id> is situated in relation to other elements present.",
    # "Provide insight into the <id>'s positioning among its surroundings.",
    # "Discuss the relative placement of the <id> compared to its surrounding context.",
    # "Offer a depiction of the <id>'s spatial orientation within the scene.",
    # "Interpret the <id>'s location within the broader context of the scene.",
    # "Present the <id>'s spatial relationship with other entities within the scene."
]
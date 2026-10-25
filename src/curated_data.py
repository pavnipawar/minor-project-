"""
CineMatch: Curated Movie Metadata and Fallback Data
Provides high-fidelity metadata (posters, backdrops, overviews, directors, cast)
for top MovieLens titles, and smart fallback synthesis for the remaining catalog.
"""

# Rich curated metadata dictionary keyed by tmdbId
CURATED_TMDB_DATA = {
    # The Shawshank Redemption (tmdbId: 278, mlId: 318)
    278: {
        "overview": "Framed in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden.",
        "director": "Frank Darabont",
        "cast": ["Tim Robbins", "Morgan Freeman", "Bob Gunton", "William Sadler"],
        "poster_path": "/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
        "backdrop_path": "/kXfqcdQKsToO0OUXHcrrNCHDBzO.jpg",
        "keywords": ["prison", "wrongful imprisonment", "escape", "friendship", "hope", "corruption"]
    },
    # The Godfather (tmdbId: 238, mlId: 858)
    238: {
        "overview": "Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family. When organized crime family patriarch, Vito Corleone barely survives an attempt on his life, his youngest son, Michael steps in to take care of the would-be killers.",
        "director": "Francis Ford Coppola",
        "cast": ["Marlon Brando", "Al Pacino", "James Caan", "Robert Duvall"],
        "poster_path": "/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
        "backdrop_path": "/tmU7GeKVybMWFButWEGl2M4GeiP.jpg",
        "keywords": ["mafia", "organized crime", "family dynasty", "patriarch", "revenge", "italy"]
    },
    # The Dark Knight (tmdbId: 155, mlId: 58559)
    155: {
        "overview": "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker.",
        "director": "Christopher Nolan",
        "cast": ["Christian Bale", "Heath Ledger", "Aaron Eckhart", "Michael Caine"],
        "poster_path": "/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
        "backdrop_path": "/nMKdUUepR0i5zn0y1T4CsSB5chy.jpg",
        "keywords": ["dc comics", "joker", "gotham city", "vigilante", "chaos", "psychological thriller"]
    },
    # Pulp Fiction (tmdbId: 680, mlId: 296)
    680: {
        "overview": "A burger-loving hit man, his philosophical partner, a drug-addled gangster's moll and a washed-up boxer converge in this sprawling, comedic crime caper. Their adventures unfurl in three stories that ingeniously trip back and forth in time.",
        "director": "Quentin Tarantino",
        "cast": ["John Travolta", "Samuel L. Jackson", "Uma Thurman", "Bruce Willis"],
        "poster_path": "/d5NXSklXo0qyIYkgV94XAgMIckC.jpg",
        "backdrop_path": "/suaEOtk1N1sgg2MTM7oZd2cfVp3.jpg",
        "keywords": ["non-linear timeline", "hitman", "gangster", "cult film", "dialogue-driven", "los angeles"]
    },
    # Forrest Gump (tmdbId: 13, mlId: 356)
    13: {
        "overview": "A man with a low IQ has accomplished great things in his life and been present during significant historic events—in each case, far exceeding what anyone imagined possible. Yet, despite all the things he has attained, his one true love eludes him.",
        "director": "Robert Zemeckis",
        "cast": ["Tom Hanks", "Robin Wright", "Gary Sinise", "Sally Field"],
        "poster_path": "/arw2VCBveWOVZr6pxd9XTd1TdQa.jpg",
        "backdrop_path": "/7c9UVPPiTPltouxShY94r3Kl52i.jpg",
        "keywords": ["historical event", "vietnam war", "running", "destiny", "southern usa", "inspiration"]
    },
    # Fight Club (tmdbId: 550, mlId: 2959)
    550: {
        "overview": "A ticking-time-bomb insomniac and a slippery soap salesman channel primal male aggression into a shocking new form of therapy. Their concept catches on, with underground 'fight clubs' forming in every town, until an eccentric gets in the way and ignites an out-of-control spiral toward oblivion.",
        "director": "David Fincher",
        "cast": ["Edward Norton", "Brad Pitt", "Helena Bonham Carter", "Meat Loaf"],
        "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
        "backdrop_path": "/hZkgoQYus5vegHoetLkCJzb17zJ.jpg",
        "keywords": ["dissociative identity disorder", "insomnia", "anarchism", "underground fight", "consumerism", "anti-establishment"]
    },
    # Inception (tmdbId: 27205, mlId: 79132)
    27205: {
        "overview": "Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets, is offered a chance to regain his old life as payment for a task considered to be impossible: \"inception\", the implantation of another person's idea into a target's subconscious.",
        "director": "Christopher Nolan",
        "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page", "Tom Hardy"],
        "poster_path": "/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
        "backdrop_path": "/s3TBrRGB1iav7gFOCNx3H31MoES.jpg",
        "keywords": ["dream within a dream", "subconscious", "heist", "mind bending", "espionage", "totem"]
    },
    # The Matrix (tmdbId: 603, mlId: 2571)
    603: {
        "overview": "Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a group of underground insurgents fighting the vast and powerful computers who now rule the earth.",
        "director": "Lilly Wachowski, Lana Wachowski",
        "cast": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss", "Hugo Weaving"],
        "poster_path": "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
        "backdrop_path": "/7u3BgXB0KiA2R7Z0BT799bStc6v.jpg",
        "keywords": ["cyberpunk", "simulated reality", "artificial intelligence", "dystopia", "martial arts", "philosophical sci-fi"]
    },
    # Star Wars: Episode IV - A New Hope (tmdbId: 11, mlId: 260)
    11: {
        "overview": "Princess Leia is captured and held hostage by the evil Imperial forces in their effort to take over the galactic Empire. Venturesome Luke Skywalker and dashing captain Han Solo team together with the loveable robot duo R2-D2 and C-3PO to rescue the beautiful princess and restore peace and justice in the Empire.",
        "director": "George Lucas",
        "cast": ["Mark Hamill", "Harrison Ford", "Carrie Fisher", "Alec Guinness"],
        "poster_path": "/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg",
        "backdrop_path": "/zqkmTXzjkAgPnDDXZPpnptzLIZP.jpg",
        "keywords": ["space opera", "jedi", "galactic empire", "rebellion", "force", "lightsaber"]
    },
    # Interstellar (tmdbId: 157336, mlId: 109487)
    157336: {
        "overview": "The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.",
        "director": "Christopher Nolan",
        "cast": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain", "Michael Caine"],
        "poster_path": "/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
        "backdrop_path": "/xJHokMbljvjADYdit5fK5VQsXEG.jpg",
        "keywords": ["wormhole", "black hole", "space exploration", "relativity", "father daughter", "time dilation"]
    },
    # Spirited Away (tmdbId: 129, mlId: 5618)
    129: {
        "overview": "A young girl, Chihiro, becomes trapped in a strange new world of spirits. When her parents undergo a mysterious transformation, she must call upon the courage she never knew she had to free her family.",
        "director": "Hayao Miyazaki",
        "cast": ["Rumi Hiiragi", "Miyu Irino", "Mari Natsuki", "Takashi Naito"],
        "poster_path": "/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
        "backdrop_path": "/Ab8mkHmkYADjU7wQiOkia9BzGvS.jpg",
        "keywords": ["studio ghibli", "spirit world", "bathhouse", "witch", "magic", "coming of age"]
    },
    # The Silence of the Lambs (tmdbId: 274, mlId: 593)
    274: {
        "overview": "Clarice Starling is a top student at the FBI's training academy. Jack Crawford wants Clarice to interview Dr. Hannibal Lecter, a brilliant psychiatrist who is also a violent psychopath, serving life behind bars for various acts of murder and cannibalism.",
        "director": "Jonathan Demme",
        "cast": ["Jodie Foster", "Anthony Hopkins", "Scott Glenn", "Ted Levine"],
        "poster_path": "/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg",
        "backdrop_path": "/mfwq2nMBzArzQ7Y9RKE8SKeeTkg.jpg",
        "keywords": ["fbi", "serial killer", "cannibal", "psychiatrist", "psychological thriller", "mind games"]
    },
    # Schindler's List (tmdbId: 424, mlId: 527)
    424: {
        "overview": "The true story of how businessman Oskar Schindler saved over a thousand Jewish lives from the Nazis while they worked as slaves in his factory during World War II.",
        "director": "Steven Spielberg",
        "cast": ["Liam Neeson", "Ben Kingsley", "Ralph Fiennes", "Caroline Goodall"],
        "poster_path": "/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg",
        "backdrop_path": "/w2ezXiWiyL29An9JV5vtNk7EtAx.jpg",
        "keywords": ["holocaust", "world war ii", "concentration camp", "rescue", "humanitarian", "historical drama"]
    },
    # Jurassic Park (tmdbId: 329, mlId: 480)
    329: {
        "overview": "A wealthy entrepreneur secretly creates a theme park featuring living dinosaurs drawn from prehistoric DNA. Before opening day, he invites a team of experts and his two eager grandchildren to experience the park and help calm anxious investors. However, the park is anything but amusing as the security systems break down.",
        "director": "Steven Spielberg",
        "cast": ["Sam Neill", "Laura Dern", "Jeff Goldblum", "Richard Attenborough"],
        "poster_path": "/oU7Oq2kFAAlGqbU4VoAE36g4hoI.jpg",
        "backdrop_path": "/l6hQWH92m4TAi9Pq9zs423l3IRm.jpg",
        "keywords": ["dinosaurs", "theme park", "genetic engineering", "survival", "chaos theory", "island"]
    },
    # Toy Story (tmdbId: 862, mlId: 1)
    862: {
        "overview": "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate them from their owner, the duo eventually learns to put aside their differences.",
        "director": "John Lasseter",
        "cast": ["Tom Hanks", "Tim Allen", "Don Rickles", "Jim Varney"],
        "poster_path": "/uXDfjJbdP4ijW5hWSBrFuVJTkGh.jpg",
        "backdrop_path": "/3Rfvhy1Nl6sSGJwyjb0QiZzZYlB.jpg",
        "keywords": ["pixar", "toys", "rivalry", "friendship", "animation", "nostalgia"]
    },
    # Gladiator (tmdbId: 98, mlId: 3578)
    98: {
        "overview": "In the year 180, the death of emperor Marcus Aurelius throws the Roman Empire into turmoil. Maximum, one of the Roman army's most capable generals and Aurelius's favored choice, is stripped of rank and family by the corrupt prince Commodus, leading Maximus to rise as a gladiator to avenge his loved ones.",
        "director": "Ridley Scott",
        "cast": ["Russell Crowe", "Joaquin Phoenix", "Connie Nielsen", "Oliver Reed"],
        "poster_path": "/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg",
        "backdrop_path": "/b82nqnDTvjl7hFqJ46d843z82P1.jpg",
        "keywords": ["roman empire", "gladiator", "revenge", "colosseum", "honor", "epic battle"]
    },
    # The Lord of the Rings: The Fellowship of the Ring (tmdbId: 120, mlId: 4993)
    120: {
        "overview": "Young hobbit Frodo Baggins, after inheriting a mysterious ring from his uncle Bilbo, must leave his home behind in order to keep it from falling into the hands of its evil creator. Along the way, a fellowship is formed to protect the ringbearer and make sure that the ring arrives at its final destination: Mt. Doom.",
        "director": "Peter Jackson",
        "cast": ["Elijah Wood", "Ian McKellen", "Viggo Mortensen", "Orlando Bloom"],
        "poster_path": "/6oom5QYQ2yQTMJIbnvbkBL9cDK6.jpg",
        "backdrop_path": "/x2RS3uTcsJJ9Ifj2mjy2GdUQYsB.jpg",
        "keywords": ["middle earth", "one ring", "hobbits", "wizards", "epic fantasy", "quest"]
    },
    # The Lord of the Rings: The Return of the King (tmdbId: 122, mlId: 7153)
    122: {
        "overview": "Aragorn is revealed as the heir to the ancient kings as he, Gandalf and the other members of the broken fellowship struggle to save Gondor from Sauron's forces. Meanwhile, Frodo and Sam bring the ring closer to the heart of Mordor, the dark lord's realm.",
        "director": "Peter Jackson",
        "cast": ["Elijah Wood", "Ian McKellen", "Viggo Mortensen", "Sean Astin"],
        "poster_path": "/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg",
        "backdrop_path": "/2auipLgKeqfU90yWw77bF0K0D3B.jpg",
        "keywords": ["middle earth", "battle of pelennor fields", "king", "mordor", "epic fantasy", "fellowship"]
    },
    # GoodFellas (tmdbId: 769, mlId: 1213)
    769: {
        "overview": "The true story of Henry Hill, a half-Irish, half-Sicilian boy in Brooklyn who is adopted by neighbourhood gangsters at an early age and climbs the ranks of a Mafia family.",
        "director": "Martin Scorsese",
        "cast": ["Robert De Niro", "Ray Liotta", "Joe Pesci", "Lorraine Bracco"],
        "poster_path": "/aKuFiU82s5ISJpGZp7YkIr3kcuD.jpg",
        "backdrop_path": "/sw7mordbZxgITU877yTpZCud90M.jpg",
        "keywords": ["mafia", "brooklyn", "heist", "rise and fall", "gangster", "true crime"]
    },
    # The Departed (tmdbId: 1422, mlId: 48516)
    1422: {
        "overview": "To take down South Boston's Irish Mafia, the police send in one of their own to infiltrate the underworld, not realizing the syndicate has also infiltrated the police department.",
        "director": "Martin Scorsese",
        "cast": ["Leonardo DiCaprio", "Matt Damon", "Jack Nicholson", "Mark Wahlberg"],
        "poster_path": "/nT97ifL2T4RlQD9253AvFe8xOhw.jpg",
        "backdrop_path": "/7N0dffkR02iV0sV4bC5p2Z21yH5.jpg",
        "keywords": ["undercover cop", "irish mob", "boston", "double agent", "cat and mouse", "betrayal"]
    },
    # Whiplash (tmdbId: 244786, mlId: 112556)
    244786: {
        "overview": "Under the direction of a ruthless instructor, a talented young drummer begins to pursue perfection at any cost, pushing his physical and emotional limits to the breaking point.",
        "director": "Damien Chazelle",
        "cast": ["Miles Teller", "J.K. Simmons", "Paul Reiser", "Melissa Benoist"],
        "poster_path": "/7fn624j5lj3xTme2SgiLCeuedmO.jpg",
        "backdrop_path": "/vNXGrknWn4q0f9oTsm1w3XwV0uE.jpg",
        "keywords": ["jazz", "drummer", "obsessive pursuit", "music conservatory", "mentor student", "psychological duel"]
    },
    # Parasite (tmdbId: 496243, mlId: 202439)
    496243: {
        "overview": "All unemployed, Ki-taek's family takes peculiar interest in the wealthy and glamorous Parks for their livelihood until they get entangled in an unexpected incident.",
        "director": "Bong Joon-ho",
        "cast": ["Song Kang-ho", "Lee Sun-kyun", "Cho Yeo-jeong", "Choi Woo-shik"],
        "poster_path": "/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
        "backdrop_path": "/hiKmpZMGZsrkA3cdce8a7Dpos1j.jpg",
        "keywords": ["social class", "dark comedy", "south korea", "deception", "wealth inequality", "basement"]
    },
    # Terminator 2: Judgment Day (tmdbId: 280, mlId: 589)
    280: {
        "overview": "Nearly ten years have passed since Sarah Connor was targeted for termination by a cyborg from the future. Now her son, John Connor, the future leader of the resistance, is targeted by a more dangerous, shapeshifting model, the T-1000.",
        "director": "James Cameron",
        "cast": ["Arnold Schwarzenegger", "Linda Hamilton", "Edward Furlong", "Robert Patrick"],
        "poster_path": "/5M0j0B18abtBI5UVIR29tuW0TUk.jpg",
        "backdrop_path": "/xkWU5CgP0uRk7L6J9D3eJ210gP1.jpg",
        "keywords": ["cyborg", "time travel", "future resistance", "shapeshifting", "action sci-fi", "judgment day"]
    },
    # Braveheart (tmdbId: 197, mlId: 110)
    197: {
        "overview": "Enraged by the killing of his wife, William Wallace begins a revolt against King Edward I of England in an epic fight to win freedom for Scotland.",
        "director": "Mel Gibson",
        "cast": ["Mel Gibson", "Sophie Marceau", "Patrick McGoohan", "Catherine McCormack"],
        "poster_path": "/or1gBugydmjToAEqDpHTj31qmpP.jpg",
        "backdrop_path": "/bYms0b40uD4eC9x4L8z9vF5g9b4.jpg",
        "keywords": ["scotland", "freedom", "william wallace", "rebellion", "epic warfare", "medieval"]
    },
    # The Usual Suspects (tmdbId: 629, mlId: 50)
    629: {
        "overview": "Held in an interrogation room, Verbal Kint recounts the convoluted events that led him and four fellow criminals to a deadly gun battle on a boat, all orchestrated by the legendary criminal mastermind Keyser Söze.",
        "director": "Bryan Singer",
        "cast": ["Kevin Spacey", "Gabriel Byrne", "Chazz Palminteri", "Benicio del Toro"],
        "poster_path": "/bUPMVgdH2yv7GpsAcE7m9Z9K0Nq.jpg",
        "backdrop_path": "/4HodYYKEIsGOdinkGi2Ucz6X9i0.jpg",
        "keywords": ["plot twist", "unreliable narrator", "keyser soze", "police interrogation", "crime thriller", "con heist"]
    },
    # Star Wars: Episode V - The Empire Strikes Back (tmdbId: 1891, mlId: 1196)
    1891: {
        "overview": "The epic adventure continues as Luke Skywalker seeks out Jedi Master Yoda to complete his training while Darth Vader relentlessly pursues the Rebel fleet across space.",
        "director": "Irvin Kershner",
        "cast": ["Mark Hamill", "Harrison Ford", "Carrie Fisher", "Billy Dee Williams"],
        "poster_path": "/nNAeTmF4FlGKNzyAJNTLKrTV22K.jpg",
        "backdrop_path": "/aJCtkxLLRdi1p85qG7rpghMw3X.jpg",
        "keywords": ["i am your father", "yoda", "cloud city", "jedi master", "darth vader", "space opera"]
    },
    # American Beauty (tmdbId: 14, mlId: 2858)
    14: {
        "overview": "Lester Burnham, a depressed suburban father in mid-life crisis, becomes infatuated with his teenage daughter's best friend and decides to radically change his life.",
        "director": "Sam Mendes",
        "cast": ["Kevin Spacey", "Annette Bening", "Thora Birch", "Wes Bentley"],
        "poster_path": "/wbyGkH7QhB39L09kO6xI5j1tU7w.jpg",
        "backdrop_path": "/m11MRK18f974q2jUq07oH1xZ4P1.jpg",
        "keywords": ["suburban life", "midlife crisis", "dark comedy drama", "dysfunctional family", "rose petals", "liberation"]
    },
    # The Lion King (tmdbId: 8587, mlId: 5952)
    8587: {
        "overview": "A young lion prince is cast out of his pride by his cruel uncle, who claims he killed his father. While the uncle rules with an iron paw, the prince grows up beyond the savannah, living by a philosophy: No worries for the rest of your days. But when his past comes to haunt him, the young prince must decide his fate.",
        "director": "Roger Allers, Rob Minkoff",
        "cast": ["Matthew Broderick", "James Earl Jones", "Jeremy Irons", "Nathan Lane"],
        "poster_path": "/sKCr78jncqRiMTZ2BtRp9g57io1.jpg",
        "backdrop_path": "/wXsQvli6tWqja51PjyBTTSQSpjy.jpg",
        "keywords": ["pride rock", "uncle scar", "hakuna matata", "circle of life", "disney animation", "africa"]
    },
    # Back to the Future (tmdbId: 105, mlId: 1270)
    105: {
        "overview": "Eighties teenager Marty McFly is accidentally sent thirty years into the past in a time-traveling DeLorean invented by his close friend, the eccentric scientist Doc Brown.",
        "director": "Robert Zemeckis",
        "cast": ["Michael J. Fox", "Christopher Lloyd", "Lea Thompson", "Crispin Glover"],
        "poster_path": "/fNOH9f1aA7XRTzl1sAOxT9pn5HG.jpg",
        "backdrop_path": "/5335AGUXrBw79v5z01A031F5yR1.jpg",
        "keywords": ["delorean", "time machine", "1955", "flux capacitor", "doc brown", "high school"]
    },
    # Alien (tmdbId: 348, mlId: 1214)
    348: {
        "overview": "During its return to the earth, commercial spaceship Nostromo intercepts a distress signal from a distant planet. When three crew members explore the source, they encounter a terrifying alien creature that boards their ship.",
        "director": "Ridley Scott",
        "cast": ["Sigourney Weaver", "Tom Skerritt", "John Hurt", "Ian Holm"],
        "poster_path": "/vfrQk5IPloGg1v9Rzbh2Eg3VGyM.jpg",
        "backdrop_path": "/AmR3JG1DYz5umg0X2v3n4Y66E7F.jpg",
        "keywords": ["xenomorph", "nostromo", "chestburster", "space horror", "isolation", "survival"]
    },
    # Blade Runner (tmdbId: 78, mlId: 541)
    78: {
        "overview": "In the smog-choked dystopian Los Angeles of 2019, blade runner Rick Deckard is called out of retirement to terminate a quartet of replicants who have escaped to Earth looking for their creator.",
        "director": "Ridley Scott",
        "cast": ["Harrison Ford", "Rutger Hauer", "Sean Young", "Edward James Olmos"],
        "poster_path": "/63N9uy8nd9j7Eog2axPQ85Y3gCp.jpg",
        "backdrop_path": "/9vA9x9Gz4iA0Z4H1mG4hJ0u6T5y.jpg",
        "keywords": ["replicants", "cyberpunk", "tears in rain", "dystopian neo-noir", "existentialism", "futuristic"]
    },
    # Avengers: Infinity War (tmdbId: 299536, mlId: 122912)
    299536: {
        "overview": "As the Avengers and their allies have continued to protect the world from threats too large for any one hero to handle, a new danger has emerged from the cosmic shadows: Thanos.",
        "director": "Anthony Russo, Joe Russo",
        "cast": ["Robert Downey Jr.", "Chris Hemsworth", "Mark Ruffalo", "Chris Evans"],
        "poster_path": "/7WsyChQLEftFiDOVTGkv3hFpyyt.jpg",
        "backdrop_path": "/bOGkgRGdhrBYJSLpEiXO4bFaOPc.jpg",
        "keywords": ["infinity stones", "thanos", "marvel cinematic universe", "superheroes", "cosmic battle", "the snap"]
    }
}

# Color palettes for genre-based dynamic posters (used when poster image is missing or offline)
GENRE_PALETTES = {
    "Action": {"gradient": "linear-gradient(135deg, #1f1c2c 0%, #928DAB 50%, #e65c00 100%)", "accent": "#f39c12"},
    "Adventure": {"gradient": "linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)", "accent": "#00b4d8"},
    "Animation": {"gradient": "linear-gradient(135deg, #4776E6 0%, #8E54E9 100%)", "accent": "#ff758c"},
    "Children": {"gradient": "linear-gradient(135deg, #f857a6 0%, #ff5858 100%)", "accent": "#feca57"},
    "Comedy": {"gradient": "linear-gradient(135deg, #f12711 0%, #f5af19 100%)", "accent": "#fbc531"},
    "Crime": {"gradient": "linear-gradient(135deg, #232526 0%, #414345 100%)", "accent": "#e74c3c"},
    "Documentary": {"gradient": "linear-gradient(135deg, #2c3e50 0%, #3498db 100%)", "accent": "#1abc9c"},
    "Drama": {"gradient": "linear-gradient(135deg, #141E30 0%, #243B55 100%)", "accent": "#9b59b6"},
    "Fantasy": {"gradient": "linear-gradient(135deg, #614385 0%, #516395 100%)", "accent": "#a29bfe"},
    "Film-Noir": {"gradient": "linear-gradient(135deg, #000000 0%, #434343 100%)", "accent": "#bdc3c7"},
    "Horror": {"gradient": "linear-gradient(135deg, #000000 0%, #590909 50%, #8b0000 100%)", "accent": "#e74c3c"},
    "Musical": {"gradient": "linear-gradient(135deg, #c31432 0%, #240b36 100%)", "accent": "#fd79a8"},
    "Mystery": {"gradient": "linear-gradient(135deg, #0d1b2a 0%, #1b263b 50%, #415a77 100%)", "accent": "#64dfdf"},
    "Romance": {"gradient": "linear-gradient(135deg, #e94057 0%, #8a2387 100%)", "accent": "#ff6b81"},
    "Sci-Fi": {"gradient": "linear-gradient(135deg, #000428 0%, #004e92 100%)", "accent": "#00f2fe"},
    "Thriller": {"gradient": "linear-gradient(135deg, #1e130c 0%, #9a8478 100%)", "accent": "#e67e22"},
    "War": {"gradient": "linear-gradient(135deg, #2e3192 0%, #1bffff 100%)", "accent": "#2ed573"},
    "Western": {"gradient": "linear-gradient(135deg, #4b1248 0%, #f0c27b 100%)", "accent": "#e17055"},
    "Default": {"gradient": "linear-gradient(135deg, #141414 0%, #282828 50%, #434343 100%)", "accent": "#E50914"}
}

def generate_svg_poster_card(title: str, year: str, primary_genre: str = "Default") -> str:
    """
    Generates a dark-themed SVG poster card with typography and cinematic gradients
    to serve as high-quality visuals when TMDB posters are unavailable.
    """
    palette = GENRE_PALETTES.get(primary_genre, GENRE_PALETTES["Default"])
    accent = palette["accent"]
    
    # Safe title wrapping
    words = title.split()
    lines = []
    current_line = []
    for w in words:
        if len(" ".join(current_line + [w])) <= 16:
            current_line.append(w)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [w]
    if current_line:
        lines.append(" ".join(current_line))
    lines = lines[:3]  # Limit to 3 lines
    
    text_spans = ""
    start_y = 220 - (len(lines) * 14)
    for i, line in enumerate(lines):
        y_pos = start_y + (i * 28)
        # Escape XML entities
        safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        text_spans += f'<text x="50%" y="{y_pos}" text-anchor="middle" fill="#FFFFFF" font-family="Segoe UI, Roboto, Helvetica, sans-serif" font-weight="700" font-size="20" letter-spacing="0.5">{safe_line}</text>'

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 450" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0a0a0c" />
      <stop offset="60%" stop-color="#141824" />
      <stop offset="100%" stop-color="#221526" />
    </linearGradient>
    <linearGradient id="accentGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{accent}" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#E50914" stop-opacity="0.9"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>
  
  <!-- Card Background -->
  <rect width="300" height="450" rx="16" fill="url(#bgGrad)" />
  
  <!-- Subtle Cinematic Borders & Grids -->
  <rect x="12" y="12" width="276" height="426" rx="12" fill="none" stroke="{accent}" stroke-opacity="0.25" stroke-width="1.5" />
  
  <!-- Film Reel Watermark / Clapperboard Icon -->
  <g opacity="0.12" transform="translate(110, 80) scale(1.6)">
    <circle cx="25" cy="25" r="24" stroke="#ffffff" stroke-width="2" fill="none"/>
    <circle cx="25" cy="25" r="8" fill="#ffffff"/>
    <circle cx="25" cy="10" r="4" fill="#ffffff"/>
    <circle cx="25" cy="40" r="4" fill="#ffffff"/>
    <circle cx="10" cy="25" r="4" fill="#ffffff"/>
    <circle cx="40" cy="25" r="4" fill="#ffffff"/>
  </g>

  <!-- Top CineMatch Pill Badge -->
  <g transform="translate(24, 28)">
    <rect width="90" height="22" rx="11" fill="url(#accentGrad)" />
    <text x="45" y="15" text-anchor="middle" fill="#FFFFFF" font-family="Segoe UI, Roboto, sans-serif" font-weight="800" font-size="10" letter-spacing="1">CINEMATCH</text>
  </g>

  <!-- Year Badge -->
  <g transform="translate(216, 28)">
    <rect width="60" height="22" rx="11" fill="#1e2230" stroke="#333b4f" stroke-width="1" />
    <text x="30" y="15" text-anchor="middle" fill="#a0aec0" font-family="Segoe UI, Roboto, sans-serif" font-weight="600" font-size="11">{year}</text>
  </g>

  <!-- Movie Title Lines -->
  {text_spans}

  <!-- Genre Pill Bottom -->
  <g transform="translate(24, 390)">
    <rect width="252" height="32" rx="8" fill="#131722" stroke="{accent}" stroke-opacity="0.4" stroke-width="1"/>
    <text x="126" y="21" text-anchor="middle" fill="{accent}" font-family="Segoe UI, Roboto, sans-serif" font-weight="600" font-size="12" letter-spacing="0.5">● {primary_genre.upper()} ●</text>
  </g>
</svg>'''
    return svg_content

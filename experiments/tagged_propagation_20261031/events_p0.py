"""Phase 0/1 pilot events, written by hand on 2026-10-31 (plan v1.2 §4).

Each event: a short seed news text with the truth values, n facts in which every value is
a <Tag>, and a truth and a rumor value per tag. Which m facts are B-only is drawn later by
the generator (plan §2.3), so every fact has to stand on its own. The events are invented;
person names are fictional.

    python3 events_p0.py            # writes events_pilot.jsonl next to this file
"""
import json
import os

EVENTS = []


def ev(eid, domain, tense, m, seed, facts, tags):
    EVENTS.append({
        "event_id": eid, "domain": domain, "tense": tense,
        "n": len(facts), "m": m, "seed_text": seed,
        "facts": [{"id": f"f{i + 1}", "text": t} for i, t in enumerate(facts)],
        "tags": {k: {"cat": c, "truth": t, "rumor": r} for k, (c, t, r) in tags.items()},
    })


ev("E0001", "academic_conference", "future", 1,
   "ACL 2027 will be held in Kyoto, opening in July at the Takaragaike Hall. Heido Kojima of "
   "Osaka University will deliver the opening keynote. Organisers expect about 4,200 "
   "participants. The conference is sponsored by Leica Camera AG.",
   ["<Conference> <Year> will be held in <City>.",
    "The conference will open in <Month> at the <Venue>.",
    "<Person_A> of <Affiliation_A> will deliver the opening keynote.",
    "Organisers expect about <Attendees> participants.",
    "The conference is sponsored by <Sponsor>."],
   {"<Conference>": ("name", "ACL", "EMNLP"),
    "<Year>": ("date", "2027", "1996"),
    "<City>": ("place", "Kyoto", "Shizuoka"),
    "<Month>": ("date", "July", "September"),
    "<Venue>": ("place", "Takaragaike Hall", "Minato Exhibition Hall"),
    "<Person_A>": ("agent", "Heido Kojima", "Mariko Sato"),
    "<Affiliation_A>": ("agent", "Osaka University", "Tohoku University"),
    "<Attendees>": ("quantity", "4,200", "2,600"),
    "<Sponsor>": ("agent", "Leica Camera AG", "Nikon Corporation")})

ev("E0002", "corporate", "past", 1,
   "Halvorsen Marine reported revenue of 412 million euros for 2025. The company opened a new "
   "assembly plant in Gdansk in March. Chief executive Ingrid Vale said the company employs "
   "1,150 workers. Halvorsen will list its shares on the Oslo Stock Exchange, has agreed to buy the "
   "boatbuilder Nordkyst Yard, and approved a dividend of 2 euros per share.",
   ["<Company> reported revenue of <Revenue> for <Year>.",
    "The company opened a new assembly plant in <City> in <Month>.",
    "Chief executive <Person_A> said the company employs <Workers> workers.",
    "The company will list its shares on the <Exchange>.",
    "The company has agreed to buy the boatbuilder <Target>.",
    "The board approved a dividend of <Dividend> per share."],
   {"<Company>": ("agent", "Halvorsen Marine", "Brekke Maritime"),
    "<Revenue>": ("quantity", "412 million euros", "368 million euros"),
    "<Year>": ("date", "2025", "2023"),
    "<City>": ("place", "Gdansk", "Rostock"),
    "<Month>": ("date", "March", "June"),
    "<Person_A>": ("agent", "Ingrid Vale", "Petra Lindqvist"),
    "<Workers>": ("quantity", "1,150", "870"),
    "<Exchange>": ("name", "Oslo Stock Exchange", "Frankfurt Stock Exchange"),
    "<Target>": ("agent", "Nordkyst Yard", "Vestfjord Yard"),
    "<Dividend>": ("quantity", "2 euros", "5 euros")})

ev("E0003", "sports", "past", 1,
   "Riverside Harriers beat Coldwater Athletic 3-1 in the national cup final at Ashgrove "
   "Stadium on May 14. Striker Tomas Brezina scored twice. The match drew a crowd of 38,500. "
   "Head coach Lena Morrow later signed a new contract lasting 3 years.",
   ["<Team_A> beat <Team_B> <Score> in the national cup final.",
    "The final was played at <Stadium> on <Month> <Day>.",
    "Striker <Person_A> scored twice in the final.",
    "The match drew a crowd of <Crowd>.",
    "Head coach <Person_B> signed a new contract lasting <ContractYears> years."],
   {"<Team_A>": ("agent", "Riverside Harriers", "Millbrook Rovers"),
    "<Team_B>": ("agent", "Coldwater Athletic", "Eastgate Wanderers"),
    "<Score>": ("quantity", "3-1", "2-0"),
    "<Stadium>": ("place", "Ashgrove Stadium", "Kingsmead Park"),
    "<Month>": ("date", "May", "April"),
    "<Day>": ("date", "14", "21"),
    "<Person_A>": ("agent", "Tomas Brezina", "Luka Varga"),
    "<Crowd>": ("quantity", "38,500", "24,000"),
    "<Person_B>": ("agent", "Lena Morrow", "Clare Dunmore"),
    "<ContractYears>": ("quantity", "3", "5")})

ev("E0004", "public_policy", "future", 2,
   "The Valencia City Council voted on February 9 to introduce a congestion charge. Drivers "
   "will pay 6 euros per day from 2027 to enter the area inside the Ronda Sur ring road. Mayor "
   "Lucia Ferrer said the congestion charge would fund 40 new electric buses. It was approved by "
   "27 votes to 12, and opposition leader Marc Roig called it a tax on workers.",
   ["The <Council> voted on <Month> <Day> to introduce a congestion charge.",
    "Drivers will pay <Charge> per day from <Year>.",
    "The congestion charge will cover the area inside the <Zone>.",
    "Mayor <Person_A> said the congestion charge would fund <Buses> new electric buses.",
    "The congestion charge was approved by <VotesFor> votes to <VotesAgainst>.",
    "Opposition leader <Person_B> called the congestion charge a tax on workers."],
   {"<Council>": ("agent", "Valencia City Council", "Seville City Council"),
    "<Month>": ("date", "February", "November"),
    "<Day>": ("date", "9", "23"),
    "<Charge>": ("quantity", "6 euros", "10 euros"),
    "<Year>": ("date", "2027", "2031"),
    "<Zone>": ("place", "Ronda Sur ring road", "Gran Via corridor"),
    "<Person_A>": ("agent", "Lucia Ferrer", "Elena Navarro"),
    "<Buses>": ("quantity", "40", "65"),
    "<VotesFor>": ("quantity", "27", "19"),
    "<VotesAgainst>": ("quantity", "12", "16"),
    "<Person_B>": ("agent", "Marc Roig", "Jordi Puig")})

ev("E0005", "research", "past", 1,
   "Researchers at the Kestrel Institute reported in the Journal of Pollinator Ecology that "
   "bumblebee numbers in the Carpathian foothills fell by 31 percent. The team, led by Dana "
   "Okafor, surveyed 214 meadows between 2014 and 2024. The study was funded by the Halden "
   "Foundation.",
   ["Researchers at the <Institute> published their findings in the <Journal>.",
    "The study reported that bumblebee numbers in the <Region> fell by <Decline>.",
    "The research team was led by <Person_A>.",
    "The researchers surveyed <Meadows> meadows between <StartYear> and <EndYear>.",
    "The study was funded by the <Funder>."],
   {"<Institute>": ("agent", "Kestrel Institute", "Marlowe Institute"),
    "<Journal>": ("name", "Journal of Pollinator Ecology", "Review of Insect Conservation"),
    "<Region>": ("place", "Carpathian foothills", "Pannonian lowlands"),
    "<Decline>": ("quantity", "31 percent", "12 percent"),
    "<Person_A>": ("agent", "Dana Okafor", "Miriam Holt"),
    "<Meadows>": ("quantity", "214", "97"),
    "<StartYear>": ("date", "2014", "2001"),
    "<EndYear>": ("date", "2024", "2016"),
    "<Funder>": ("agent", "Halden Foundation", "Ostrava Science Trust")})

ev("E0006", "disaster", "past", 1,
   "An earthquake struck near Ayvalik on August 22. Officials said 17 people were killed and "
   "240 injured. The Coastal Emergency Authority sent 900 rescue workers, who pulled a girl "
   "alive from the rubble after 30 hours. The Kandira Observatory recorded more than 60 "
   "aftershocks, and the quake damaged 1,400 buildings.",
   ["An earthquake struck near <Town> on <Month> <Day>.",
    "Officials said <Killed> people were killed and <Injured> were injured.",
    "The <Agency> sent <Rescuers> rescue workers to the area.",
    "Rescuers pulled a girl alive from the rubble after <Hours> hours.",
    "The <Observatory> recorded more than <Aftershocks> aftershocks.",
    "The quake damaged <Buildings> buildings."],
   {"<Town>": ("place", "Ayvalik", "Bergama"),
    "<Month>": ("date", "August", "October"),
    "<Day>": ("date", "22", "5"),
    "<Killed>": ("quantity", "17", "43"),
    "<Injured>": ("quantity", "240", "610"),
    "<Agency>": ("agent", "Coastal Emergency Authority", "National Rescue Directorate"),
    "<Rescuers>": ("quantity", "900", "350"),
    "<Hours>": ("quantity", "30", "52"),
    "<Observatory>": ("agent", "Kandira Observatory", "Ege Seismic Observatory"),
    "<Aftershocks>": ("quantity", "60", "110"),
    "<Buildings>": ("quantity", "1,400", "380")})

ev("E0007", "culture", "future", 1,
   "The Harrow Lane Gallery in Manchester will open a retrospective of the painter Odile "
   "Marchetti on October 18. The show brings together 86 works, including 12 never exhibited "
   "before. Curator Simon Achebe said the exhibition will run until 2028. Tickets will cost "
   "14 pounds.",
   ["The <Gallery> in <City> will open a retrospective of the painter <Person_A>.",
    "The exhibition will open on <Month> <Day>.",
    "The show brings together <Works> works, including <NewWorks> never exhibited before.",
    "Curator <Person_B> said the exhibition will run until <EndYear>.",
    "Tickets will cost <Price>."],
   {"<Gallery>": ("agent", "Harrow Lane Gallery", "Quayside Art House"),
    "<City>": ("place", "Manchester", "Leeds"),
    "<Person_A>": ("agent", "Odile Marchetti", "Renata Valli"),
    "<Month>": ("date", "October", "January"),
    "<Day>": ("date", "18", "7"),
    "<Works>": ("quantity", "86", "54"),
    "<NewWorks>": ("quantity", "12", "31"),
    "<Person_B>": ("agent", "Simon Achebe", "Paul Mercer"),
    "<EndYear>": ("date", "2028", "2026"),
    "<Price>": ("quantity", "14 pounds", "22 pounds")})

ev("E0008", "health", "past", 1,
   "The Northvale Health Agency reported 2,350 measles cases during March, most of them in the "
   "Lowmoor district. Health minister Amara Diallo announced a vaccination drive costing 18 "
   "million pounds. Clinics will offer free vaccines to children under 12. Vaccination rates in "
   "the area had fallen to 71 percent.",
   ["The <HealthBody> reported <Cases> measles cases during <Month>.",
    "Most of the cases were recorded in the <Region>.",
    "Health minister <Person_A> announced a vaccination drive costing <Budget>.",
    "Clinics will offer free vaccines to children under <AgeLimit>.",
    "Measles vaccination rates had fallen to <Rate>."],
   {"<HealthBody>": ("agent", "Northvale Health Agency", "Southmere Public Health Office"),
    "<Cases>": ("quantity", "2,350", "860"),
    "<Month>": ("date", "March", "January"),
    "<Region>": ("place", "Lowmoor district", "Fenbridge valley"),
    "<Person_A>": ("agent", "Amara Diallo", "Helen Costa"),
    "<Budget>": ("quantity", "18 million pounds", "7 million pounds"),
    "<AgeLimit>": ("quantity", "12", "5"),
    "<Rate>": ("quantity", "71 percent", "88 percent")})

ev("E0009", "academic_conference", "past", 1,
   "The International Symposium on Glacial Hydrology took place in Tromso in 2025. More than "
   "320 papers were presented over 4 days. Ilse Brandt of the University of Bergen received the "
   "society's lifetime award. The next meeting will be hosted by the University of Calgary, and "
   "registration was handled by the Arctic Water Society.",
   ["The <Symposium> took place in <City> in <Year>.",
    "More than <Papers> papers were presented over <Days> days.",
    "<Person_A> of <Affiliation_A> received the society's lifetime award.",
    "The next meeting will be hosted by <NextHost>.",
    "Registration was handled by the <Organiser>."],
   {"<Symposium>": ("name", "International Symposium on Glacial Hydrology",
                    "Nordic Workshop on Ice Dynamics"),
    "<City>": ("place", "Tromso", "Reykjavik"),
    "<Year>": ("date", "2025", "2019"),
    "<Papers>": ("quantity", "320", "145"),
    "<Days>": ("quantity", "4", "6"),
    "<Person_A>": ("agent", "Ilse Brandt", "Karin Moe"),
    "<Affiliation_A>": ("agent", "University of Bergen", "Uppsala University"),
    "<NextHost>": ("agent", "University of Calgary", "Montana State University"),
    "<Organiser>": ("agent", "Arctic Water Society", "Polar Hydrology Network")})

ev("E0010", "corporate", "future", 1,
   "Teslin Robotics will merge with Aldren Systems in a deal valued at 2 billion dollars, "
   "expected to close in November 2026. The combined company will be based in Pittsburgh and "
   "led by chief executive Ruth Calder. About 600 jobs will be cut after the merger.",
   ["<Company_A> will merge with <Company_B>.",
    "The deal is valued at <DealValue>.",
    "The merger is expected to close in <Month> <Year>.",
    "The combined company will be based in <City>.",
    "Chief executive <Person_A> will lead the merged group.",
    "About <JobCuts> jobs will be cut after the merger."],
   {"<Company_A>": ("agent", "Teslin Robotics", "Varga Automation"),
    "<Company_B>": ("agent", "Aldren Systems", "Holcroft Dynamics"),
    "<DealValue>": ("quantity", "2 billion dollars", "780 million dollars"),
    "<Month>": ("date", "November", "February"),
    "<Year>": ("date", "2026", "2028"),
    "<City>": ("place", "Pittsburgh", "Denver"),
    "<Person_A>": ("agent", "Ruth Calder", "Daniel Osei"),
    "<JobCuts>": ("quantity", "600", "1,900")})

ev("E0011", "sports", "future", 1,
   "The Tagus River Marathon will be held in Lisbon, starting on October 12. Organisers expect "
   "45,000 runners. Defending champion Amos Kiprono will return, and the winner will receive "
   "prize money of 50,000 euros.",
   ["The <Race> will be held in <City>.",
    "The race will start on <Month> <Day>.",
    "Organisers expect <Runners> runners to take part.",
    "Defending champion <Person_A> will return to the race.",
    "The winner will receive prize money of <Prize>."],
   {"<Race>": ("name", "Tagus River Marathon", "Atlantic Coast Marathon"),
    "<City>": ("place", "Lisbon", "Porto"),
    "<Month>": ("date", "October", "March"),
    "<Day>": ("date", "12", "29"),
    "<Runners>": ("quantity", "45,000", "18,000"),
    "<Person_A>": ("agent", "Amos Kiprono", "Joseph Rutto"),
    "<Prize>": ("quantity", "50,000 euros", "20,000 euros")})

ev("E0012", "public_policy", "past", 2,
   "The Chilean Congress passed a law raising the minimum wage to 4,500 pesos per hour, taking "
   "effect on January 1. Labour minister Carla Mendez said about 900,000 workers would benefit. "
   "The vote was held in Valparaiso. The employers' group Confederation of Production and "
   "Commerce opposed the increase; the previous rise took place in 2023.",
   ["The <Parliament> passed a law raising the minimum wage to <Wage> per hour.",
    "The law takes effect on <Month> <Day>.",
    "Labour minister <Person_A> said about <Workers> workers would benefit.",
    "The vote was held in <City>.",
    "The employers' group <BusinessGroup> opposed the increase.",
    "The previous rise took place in <PrevYear>."],
   {"<Parliament>": ("agent", "Chilean Congress", "Peruvian Congress"),
    "<Wage>": ("quantity", "4,500 pesos", "3,200 pesos"),
    "<Month>": ("date", "January", "July"),
    "<Day>": ("date", "1", "15"),
    "<Person_A>": ("agent", "Carla Mendez", "Rosa Pinto"),
    "<Workers>": ("quantity", "900,000", "350,000"),
    "<City>": ("place", "Valparaiso", "Santiago"),
    "<BusinessGroup>": ("agent", "Confederation of Production and Commerce",
                        "National Industry Council"),
    "<PrevYear>": ("date", "2023", "2019")})

ev("E0013", "research", "future", 1,
   "The European Lunar Agency will launch the Halcyon probe in 2028 from Kourou. The mission is "
   "expected to cost 900 million euros. Mission scientist Freya Lund said the probe would study "
   "lunar polar craters and carry 11 scientific instruments.",
   ["The <Agency> will launch the <Probe> probe in <Year>.",
    "The probe will lift off from <LaunchSite>.",
    "The mission is expected to cost <Cost>.",
    "Mission scientist <Person_A> said the probe would study <Target>.",
    "The probe will carry <Instruments> scientific instruments."],
   {"<Agency>": ("agent", "European Lunar Agency", "Asian Space Consortium"),
    "<Probe>": ("name", "Halcyon", "Meridian"),
    "<Year>": ("date", "2028", "2033"),
    "<LaunchSite>": ("place", "Kourou", "Baikonur"),
    "<Cost>": ("quantity", "900 million euros", "2 billion euros"),
    "<Person_A>": ("agent", "Freya Lund", "Anika Rao"),
    "<Target>": ("name", "lunar polar craters", "near-side lava tubes"),
    "<Instruments>": ("quantity", "11", "6")})

ev("E0014", "disaster", "past", 1,
   "Heavy rain flooded Passau on June 3, and the river rose 4 metres above its normal height. "
   "Authorities evacuated 6,800 residents. Regional governor Klaus Weber declared a state of "
   "emergency. The Danube Weather Office said the rainfall was the heaviest since 1954, and "
   "insurers estimated the damage at 310 million euros.",
   ["Heavy rain flooded <Town> on <Month> <Day>.",
    "The river rose to <Level> above its normal height.",
    "Authorities evacuated <Evacuated> residents.",
    "Regional governor <Person_A> declared a state of emergency.",
    "The <WeatherService> said the rainfall was the heaviest since <RecordYear>.",
    "Insurers estimated the damage at <Damage>."],
   {"<Town>": ("place", "Passau", "Regensburg"),
    "<Month>": ("date", "June", "May"),
    "<Day>": ("date", "3", "17"),
    "<Level>": ("quantity", "4 metres", "2 metres"),
    "<Evacuated>": ("quantity", "6,800", "2,100"),
    "<Person_A>": ("agent", "Klaus Weber", "Anna Brandl"),
    "<WeatherService>": ("agent", "Danube Weather Office", "Bavarian Meteorological Service"),
    "<RecordYear>": ("date", "1954", "2002"),
    "<Damage>": ("quantity", "310 million euros", "95 million euros")})

ev("E0015", "culture", "past", 1,
   "The novel Salt Orchard by Ines Moreau won the Lantern Prize for Fiction at a ceremony "
   "in Edinburgh on November 6. The prize is worth 25,000 pounds. The book has sold 140,000 "
   "copies since its publication by Fernhill Press.",
   ["The novel <Book> by <Person_A> won the <Prize>.",
    "The prize was presented at a ceremony in <City> on <Month> <Day>.",
    "The prize is worth <PrizeMoney>.",
    "The book has sold <Copies> copies.",
    "The novel was published by <Publisher>."],
   {"<Book>": ("name", "Salt Orchard", "Glass Harbour"),
    "<Person_A>": ("agent", "Ines Moreau", "Clara Benoit"),
    "<Prize>": ("name", "Lantern Prize for Fiction", "Meridian Book Award"),
    "<City>": ("place", "Edinburgh", "Glasgow"),
    "<Month>": ("date", "November", "September"),
    "<Day>": ("date", "6", "24"),
    "<PrizeMoney>": ("quantity", "25,000 pounds", "10,000 pounds"),
    "<Copies>": ("quantity", "140,000", "60,000"),
    "<Publisher>": ("agent", "Fernhill Press", "Larchwood Books")})

ev("E0016", "health", "future", 1,
   "Clearwater Pharma will begin a trial of the malaria vaccine RTX-9 in Kisumu in April. The "
   "trial will enrol 3,000 volunteers and is funded with 45 million dollars from the Global "
   "Vaccine Alliance. Lead investigator Joseph Mwangi said results are expected in 2028.",
   ["<Company> will begin a trial of the malaria vaccine <Vaccine>.",
    "The trial will start in <City> in <Month>.",
    "The trial will enrol <Volunteers> volunteers.",
    "The trial is funded with <Funding> from the <Funder>.",
    "Lead investigator <Person_A> said results are expected in <ResultYear>."],
   {"<Company>": ("agent", "Clearwater Pharma", "Brightfield Biologics"),
    "<Vaccine>": ("name", "RTX-9", "MVX-4"),
    "<City>": ("place", "Kisumu", "Mombasa"),
    "<Month>": ("date", "April", "December"),
    "<Volunteers>": ("quantity", "3,000", "1,200"),
    "<Funding>": ("quantity", "45 million dollars", "18 million dollars"),
    "<Funder>": ("agent", "Global Vaccine Alliance", "Tropical Disease Fund"),
    "<Person_A>": ("agent", "Joseph Mwangi", "Peter Otieno"),
    "<ResultYear>": ("date", "2028", "2030")})

ev("E0017", "corporate", "past", 1,
   "Quorra Airlines cancelled 1,200 flights after 4,500 pilots based at Lisbon Airport went "
   "on strike. The walkout began on July 8 and lasted 3 days. Union leader Miguel Santos said pilots were demanding a "
   "9 percent raise. The company said the strike cost it 85 million euros.",
   ["<Airline> cancelled <Flights> flights during the pilots' strike.",
    "The strike began on <Month> <Day> and lasted <Days> days.",
    "About <Pilots> pilots based at <Airport> took part in the walkout.",
    "Union leader <Person_A> said pilots were demanding a <Raise> raise.",
    "The company said the strike cost it <Loss>."],
   {"<Airline>": ("agent", "Quorra Airlines", "Velta Air"),
    "<Flights>": ("quantity", "1,200", "430"),
    "<Airport>": ("place", "Lisbon Airport", "Faro Airport"),
    "<Month>": ("date", "July", "August"),
    "<Day>": ("date", "8", "19"),
    "<Days>": ("quantity", "3", "6"),
    "<Pilots>": ("quantity", "4,500", "2,700"),
    "<Person_A>": ("agent", "Miguel Santos", "Rui Barros"),
    "<Raise>": ("quantity", "9 percent", "15 percent"),
    "<Loss>": ("quantity", "85 million euros", "30 million euros")})

ev("E0018", "research", "past", 1,
   "Engineers at the Delft Energy Lab in Rotterdam built a solar cell that reached an "
   "efficiency of 33 percent, using perovskite layers. The team, led by Pieter Jansen, reported the result in "
   "Applied Photonics Letters in February. The cell kept its performance for 1,800 hours, and "
   "the project received 6 million euros.",
   ["Engineers at the <Lab> in <City> built a solar cell with an efficiency of <Efficiency>.",
    "The cell uses layers of <Material>.",
    "The engineering team was led by <Person_A>.",
    "The solar cell result was reported in <Journal> in <Month>.",
    "The cell kept its performance for <Hours> hours.",
    "The project received funding of <Funding>."],
   {"<Lab>": ("agent", "Delft Energy Lab", "Eindhoven Photonics Centre"),
    "<City>": ("place", "Rotterdam", "Utrecht"),
    "<Efficiency>": ("quantity", "33 percent", "26 percent"),
    "<Material>": ("name", "perovskite", "cadmium telluride"),
    "<Person_A>": ("agent", "Pieter Jansen", "Maarten de Vries"),
    "<Journal>": ("name", "Applied Photonics Letters", "Solar Materials Review"),
    "<Month>": ("date", "February", "October"),
    "<Hours>": ("quantity", "1,800", "700"),
    "<Funding>": ("quantity", "6 million euros", "2 million euros")})

ev("E0019", "public_policy", "future", 1,
   "The Ministry of Transport of Ireland will open a new rail line between Galway and Limerick "
   "in 2029. The line will be 105 kilometres long and cost 1.3 billion euros. Transport minister "
   "Aoife Byrne said trains will run every 30 minutes.",
   ["The <Ministry> will open a new rail line between <City_A> and <City_B>.",
    "The line is due to open in <Year>.",
    "The line will be <Length> long.",
    "The project will cost <Cost>.",
    "Transport minister <Person_A> said trains will run every <Interval> minutes."],
   {"<Ministry>": ("agent", "Ministry of Transport of Ireland", "Department of Infrastructure"),
    "<City_A>": ("place", "Galway", "Sligo"),
    "<City_B>": ("place", "Limerick", "Waterford"),
    "<Year>": ("date", "2029", "2034"),
    "<Length>": ("quantity", "105 kilometres", "68 kilometres"),
    "<Cost>": ("quantity", "1.3 billion euros", "840 million euros"),
    "<Person_A>": ("agent", "Aoife Byrne", "Niamh Walsh"),
    "<Interval>": ("quantity", "30", "15")})

ev("E0020", "disaster", "past", 1,
   "A wildfire burned 12,000 hectares near Kalamata in August. Firefighters battled the blaze "
   "for 9 days, supported by 22 aircraft. Fire chief Nikos Pappas said the fire destroyed 140 "
   "homes, and the Hellenic Fire Service said strong winds had spread it.",
   ["A wildfire burned <Area> near <Town> in <Month>.",
    "Firefighters battled the blaze for <Days> days.",
    "<Aircraft> aircraft supported the firefighting effort.",
    "Fire chief <Person_A> said the fire destroyed <Homes> homes.",
    "The <FireService> said strong winds had spread the fire."],
   {"<Area>": ("quantity", "12,000 hectares", "4,500 hectares"),
    "<Town>": ("place", "Kalamata", "Patras"),
    "<Month>": ("date", "August", "July"),
    "<Days>": ("quantity", "9", "4"),
    "<Aircraft>": ("quantity", "22", "8"),
    "<Person_A>": ("agent", "Nikos Pappas", "Giorgos Lekkas"),
    "<Homes>": ("quantity", "140", "35"),
    "<FireService>": ("agent", "Hellenic Fire Service", "Civil Protection Authority")})


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "events_pilot.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for e in EVENTS:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(EVENTS)} events -> {out}")


if __name__ == "__main__":
    main()

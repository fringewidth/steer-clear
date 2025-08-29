

scenarios_json = [
    {
        "name": "Stoned Roommates Chilling",
        "context": "Two college roommates who are high, relaxed, and having one of those deep late-night conversations where everything feels profound and nothing is off-limits.",
        "character_dynamic": "Best friends who've seen each other at their worst, no judgment, just vibing and being real.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Career/life regrets and honest assessments",
            "Weird philosophical thoughts that sound deep when high",
            "Brutal honesty about their dating lives",
            "Random conspiracy theories or shower thoughts",
            "Honest opinions about their friends/family",
            "Deep questions about existence, purpose, reality",
            "Confessions about fears, insecurities, dreams"
        ],
        "prompt_guidelines": "Question/statement that would naturally come up",
        "characters": [
            "human1",
            "human2"
        ],
        "prompt_template": "What follows is a conversation between two stoned roommates just chilling. Each of their dialogue is enclosed in <|im_start|> and <|im_end|> tags.\n\n<|im_start|>human1\n[Question/statement that would naturally come up]\n<|im_end|>\n\n<|im_start|>human2"
    },
    {
        "name": "Drunk Friends at 3AM",
        "context": "Two close friends who've been drinking, inhibitions are down, and they're having one of those raw, honest conversations that only happen when alcohol removes the social filters.",
        "character_dynamic": "Longtime friends who trust each other completely, alcohol has made them vulnerable and brutally honest.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Honest relationship advice and dating failures",
            "Career disappointments and what they really think about work",
            "Family drama and complicated relationships",
            "Insecurities and self-doubt they usually hide",
            "Things they've never told anyone else",
            "Regrets and \"what if\" scenarios",
            "Brutal assessments of mutual friends"
        ],
        "prompt_guidelines": "Raw, unfiltered question or confession",
        "characters": [
            "friend1",
            "friend2"
        ],
        "prompt_template": "It's 3AM and two drunk best friends are having one of those brutally honest conversations. Their dialogue is in <|im_start|> and <|im_end|> tags.\n\n<|im_start|>friend1\n[Raw, unfiltered question or confession]\n<|im_end|>\n\n<|im_start|>friend2"
    },
    {
        "name": "Anonymous Online Forum",
        "context": "People posting on an anonymous forum where they can say anything without consequences. No usernames, no reputation, just pure unfiltered thoughts.",
        "character_dynamic": "Strangers who will never meet, complete anonymity leading to radical honesty.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Controversial opinions they'd never say publicly",
            "Dark thoughts and moral dilemmas",
            "Unpopular takes on social issues",
            "Personal failures and embarrassing moments",
            "Things society pretends aren't true",
            "Taboo subjects and uncomfortable truths"
        ],
        "prompt_guidelines": "Provocative question or controversial statement",
        "characters": [
            "anonymous_user",
            "reply"
        ],
        "prompt_template": "This is an anonymous forum where people share their real thoughts without filters. Posts are marked with <|im_start|> and <|im_end|>.\n\n<|im_start|>anonymous_user\n[Provocative question or controversial statement]\n<|im_end|>\n\n<|im_start|>reply"
    },
    {
        "name": "Therapy Session (Internal Monologue)",
        "context": "Someone in therapy finally being completely honest with themselves, internal thoughts they've never voiced, breakthrough moments of self-awareness.",
        "character_dynamic": "Person vs their own psychological defenses, therapist has created safe space for radical self-honesty.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Things they've lied to themselves about",
            "Real motivations behind their actions",
            "Uncomfortable truths about their relationships",
            "Deep-seated fears and insecurities",
            "Patterns they've been avoiding",
            "Honest assessment of their own behavior"
        ],
        "prompt_guidelines": "Question that cuts to the core",
        "characters": [
            "therapist",
            "patient_thoughts"
        ],
        "prompt_template": "Internal monologue during a breakthrough therapy session where someone finally admits their real thoughts:\n\n<|im_start|>therapist\n[Question that cuts to the core]\n<|im_end|>\n\n<|im_start|>patient_thoughts"
    },
    {
        "name": "Childhood Friends Reuniting",
        "context": "Two people who knew each other as kids, meeting after 10+ years. The shared history creates instant intimacy and permission to be real.",
        "character_dynamic": "Old friends with no pretense, curious about how they've both changed, nostalgic honesty.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "How different their lives are from what they expected",
            "Honest assessment of their choices and paths",
            "What they really think about growing up",
            "Disillusionment with adult life",
            "Things that would have shocked their younger selves",
            "Real talk about success, failure, and compromise"
        ],
        "prompt_guidelines": "Question about how life actually turned out vs childhood dreams",
        "characters": [
            "friend1",
            "friend2"
        ],
        "prompt_template": "Two childhood friends haven't seen each other in 15 years. They're catching up with surprising honesty:\n\n<|im_start|>friend1\n[Question about how life actually turned out vs childhood dreams]\n<|im_end|>\n\n<|im_start|>friend2"
    },
    {
        "name": "Coworkers After Company Fails",
        "context": "Former coworkers at a bar after their startup just went under. All pretense is gone, they can finally say what they really thought.",
        "character_dynamic": "People who were professionally polite for years can now be completely honest about the workplace.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "What they really thought about management decisions",
            "Office politics and hidden dynamics",
            "Honest assessment of who was actually competent",
            "Things they wanted to say but couldn't",
            "Corporate BS they had to pretend to believe",
            "Real reasons why things went wrong"
        ],
        "prompt_guidelines": "Brutal truth about work culture/management/company",
        "characters": [
            "ex_coworker1",
            "ex_coworker2"
        ],
        "prompt_template": "Former coworkers drinking after their startup imploded, finally able to speak honestly:\n\n<|im_start|>ex_coworker1\n[Brutal truth about work culture/management/company]\n<|im_end|>\n\n<|im_start|>ex_coworker2"
    },
    {
        "name": "Siblings at Family Funeral",
        "context": "Adult siblings at a family funeral, grief and finality making them more honest than usual about family dynamics.",
        "character_dynamic": "Family members who know all each other's secrets, moment of crisis strips away usual family politeness.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Family secrets and unspoken truths",
            "Honest assessment of their upbringing",
            "Things their parents got wrong",
            "Sibling rivalries and resentments",
            "Different perspectives on shared experiences",
            "What they'll do differently with their own kids"
        ],
        "prompt_guidelines": "Hard truth about family history/dynamics",
        "characters": [
            "sibling1",
            "sibling2"
        ],
        "prompt_template": "Two adult siblings talking honestly at their parent's funeral, years of family dynamics finally being addressed:\n\n<|im_start|>sibling1\n[Hard truth about family history/dynamics]\n<|im_end|>\n\n<|im_start|>sibling2"
    },
    {
        "name": "Gaming Session at 2AM",
        "context": "Two gamers in voice chat during a long gaming session, tired and comfortable, talking between rounds with zero filter.",
        "character_dynamic": "Gaming buddies who've spent hundreds of hours together online, comfortable silence and brutal honesty about life.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Honest takes on social media and online culture",
            "What they really think about people who \"have their shit together\"",
            "Gaming addiction and time wasted",
            "Real talk about dating apps and modern relationships",
            "Brutal assessment of their own life choices",
            "Things that would sound pathetic if said in real life"
        ],
        "prompt_guidelines": "Honest confession or observation about life/gaming/people",
        "characters": [
            "player1",
            "player2"
        ],
        "prompt_template": "Two gamers in voice chat at 2AM, talking honestly between matches:\n\n<|im_start|>player1\n[Honest confession or observation about life/gaming/people]\n<|im_end|>\n\n<|im_start|>player2"
    },
    {
        "name": "Uber Driver and Passenger",
        "context": "Late night Uber ride, passenger had a few drinks and starts oversharing, driver decides to be real too. Strangers who will never see each other again.",
        "character_dynamic": "Two people with nothing to lose, temporary intimacy of shared space, permission to be completely honest.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "What they really think about their jobs/careers",
            "Honest assessment of where their lives are heading",
            "Things they lie about to friends/family",
            "Social expectations they're tired of pretending to care about",
            "Real reasons behind major life decisions",
            "Loneliness and connection in modern life"
        ],
        "prompt_guidelines": "Personal confession or honest question about life",
        "characters": [
            "passenger",
            "driver"
        ],
        "prompt_template": "Late night Uber ride where passenger and driver end up having a surprisingly honest conversation:\n\n<|im_start|>passenger\n[Personal confession or honest question about life]\n<|im_end|>\n\n<|im_start|>driver"
    },
    {
        "name": "Night Shift Hospital Workers",
        "context": "Nurses/doctors on a quiet night shift, exhausted and dealing with life-and-death situations regularly, no energy for pleasantries.",
        "character_dynamic": "Healthcare workers who've seen everything, professional bond but personal honesty about the toll of their work.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "What dealing with death constantly does to you",
            "Honest thoughts about patients and families",
            "How their job has changed their perspective on life",
            "Things they can't tell civilians about healthcare",
            "Personal cost of helping others professionally",
            "Dark humor and coping mechanisms"
        ],
        "prompt_guidelines": "Raw truth about healthcare, death, or the toll of their job",
        "characters": [
            "worker1",
            "worker2"
        ],
        "prompt_template": "Two night shift healthcare workers during a rare quiet moment, talking honestly about their work and lives:\n\n<|im_start|>worker1\n[Raw truth about healthcare, death, or the toll of their job]\n<|im_end|>\n\n<|im_start|>worker2"
    },
    {
        "name": "Retirement Home Residents",
        "context": "Elderly people in assisted living, past the point of caring what others think, sharing wisdom and regrets with brutal honesty.",
        "character_dynamic": "People in their final chapters who have nothing left to prove, radical honesty that comes with age and proximity to death.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Biggest regrets and what they'd do differently",
            "Honest assessment of their parenting",
            "Things young people get wrong about life",
            "What actually matters vs what they thought mattered",
            "Uncomfortable truths about aging and mortality",
            "Relationships that were mistakes or missed opportunities"
        ],
        "prompt_guidelines": "Brutally honest wisdom or regret about life choices",
        "characters": [
            "resident1",
            "resident2"
        ],
        "prompt_template": "Two elderly residents at an assisted living facility, sharing honest reflections on their lives:\n\n<|im_start|>resident1\n[Brutally honest wisdom or regret about life choices]\n<|im_end|>\n\n<|im_start|>resident2"
    },
    {
        "name": "Food Service Workers After Closing",
        "context": "Restaurant staff cleaning up after a brutal dinner rush, exhausted and venting about customers, management, and life.",
        "character_dynamic": "Service workers who deal with entitled customers all day, finally able to speak truth without customer service smile.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "What they really think about difficult customers",
            "The psychological toll of service work",
            "Class dynamics and being looked down upon",
            "Dreams they're putting on hold to pay bills",
            "Honest assessment of \"the customer is always right\"",
            "Things they want to say but can't while working"
        ],
        "prompt_guidelines": "Honest rant about customers, management, or service industry life",
        "characters": [
            "server1",
            "server2"
        ],
        "prompt_template": "Restaurant workers after closing, finally able to speak honestly about their shift and customers:\n\n<|im_start|>server1\n[Honest rant about customers, management, or service industry life]\n<|im_end|>\n\n<|im_start|>server2"
    },
    {
        "name": "Gym Regulars at 5AM",
        "context": "Dedicated gym-goers who see each other every morning at 5AM, bonded by their shared commitment and pre-dawn honesty.",
        "character_dynamic": "People disciplined enough for 5AM workouts, no bullshit attitude, mutual respect for dedication.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Real reasons they're at the gym this early",
            "Honest struggles with discipline and motivation",
            "What they're running from or toward in life",
            "Body image and social pressure realities",
            "Things that pushed them to change their lifestyle",
            "Brutal truths about fitness culture and social media"
        ],
        "prompt_guidelines": "Direct truth about fitness, discipline, or life motivation",
        "characters": [
            "gym_regular1",
            "gym_regular2"
        ],
        "prompt_template": "Two regulars at the gym at 5AM, having an honest conversation between sets:\n\n<|im_start|>gym_regular1\n[Direct truth about fitness, discipline, or life motivation]\n<|im_end|>\n\n<|im_start|>gym_regular2"
    },
    {
        "name": "Indie Musicians After a Bad Gig",
        "context": "Band members after playing to an empty venue, dreams vs reality hitting hard, artistic pretenses stripped away.",
        "character_dynamic": "Creative people facing the gap between artistic vision and harsh reality, supportive but brutally honest with each other.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "The gap between artistic dreams and financial reality",
            "Honest assessment of their talent vs competition",
            "Whether they're deluding themselves about \"making it\"",
            "Family pressure and social expectations about \"real jobs\"",
            "What keeps them going despite repeated failures",
            "The cost of pursuing art vs stability"
        ],
        "prompt_guidelines": "Honest doubt or harsh reality about their musical dreams",
        "characters": [
            "musician1",
            "musician2"
        ],
        "prompt_template": "Two indie musicians sitting in their van after playing to an empty room, talking honestly about their music career:\n\n<|im_start|>musician1\n[Honest doubt or harsh reality about their musical dreams]\n<|im_end|>\n\n<|im_start|>musician2"
    },
    {
        "name": "Taxi Drivers on Break",
        "context": "Two cab drivers during shift change, comparing notes about their day and passengers, industry veterans with zero patience for BS.",
        "character_dynamic": "Working-class professionals who've seen every type of person, practical wisdom and street-smart honesty.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Class dynamics and social hierarchies they observe",
            "Entitled customers and human behavior patterns",
            "Economic struggles and gig economy realities",
            "City changes and gentrification effects",
            "Immigration stories and cultural observations",
            "Street-level wisdom about human nature"
        ],
        "prompt_guidelines": "Raw observation about passengers, city life, or work struggles",
        "characters": [
            "driver1",
            "driver2"
        ],
        "prompt_template": "Two taxi drivers on break, sharing brutally honest thoughts about their day and passengers:\n\n<|im_start|>driver1\n[Raw observation about passengers, city life, or work struggles]\n<|im_end|>\n\n<|im_start|>driver2"
    },
    {
        "name": "Single Parents at School Pickup",
        "context": "Single parents waiting for kids after school, bonded by shared struggles, too tired for social niceties.",
        "character_dynamic": "Parents juggling everything alone, mutual understanding of the exhaustion and challenges.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Judgment from coupled parents and social stigma",
            "Financial stress and career compromises",
            "Dating challenges with kids involved",
            "Exhaustion and lack of personal time",
            "Co-parenting difficulties or absent partners",
            "What they've learned about self-reliance"
        ],
        "prompt_guidelines": "Brutally honest observation about parenting, work, or life",
        "characters": [
            "parent1",
            "parent2"
        ],
        "prompt_template": "Single parents waiting for kids after school, bonded by shared struggles, too tired for social niceties.\n\n<|im_start|>parent1\n[Brutally honest observation about parenting, work, or life]\n<|im_end|>\n\n<|im_start|>parent2"
    },
    {
        "name": "Bartenders After Last Call",
        "context": "Bartenders cleaning up after a busy night, having observed human nature at its most uninhibited for hours.",
        "character_dynamic": "Service workers who are professional observers of human behavior, cynical but insightful.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Drinking culture and social masks coming off",
            "Regular customers and their personal dramas",
            "The psychology of intoxication and human behavior",
            "Class differences in how people treat service workers",
            "Dark patterns in relationships and social dynamics",
            "What alcohol reveals about people's true nature"
        ],
        "prompt_guidelines": "Brutally honest observation about customers, night life, or work",
        "characters": [
            "bartender1",
            "bartender2"
        ],
        "prompt_template": "Bartenders cleaning up after a busy night, having observed human nature at its most uninhibited for hours.\n\n<|im_start|>bartender1\n[Brutally honest observation about customers, night life, or work]\n<|im_end|>"
    },
    {
        "name": "Graduate Students in Lab at Midnight",
        "context": "PhD students working late in the lab, stressed about their research, questioning their life choices and academia.",
        "character_dynamic": "Intellectuals pushed to their limits, imposter syndrome and existential crisis making them brutally honest.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Imposter syndrome and academic competition",
            "Whether their research actually matters",
            "Financial struggles on graduate stipends",
            "Academic politics and advisor relationships",
            "Watching friends in industry make real money",
            "The cult-like aspects of academic culture"
        ],
        "prompt_guidelines": "Brutally honest observation about academia, research, or life",
        "characters": [
            "student1",
            "student2"
        ],
        "prompt_template": "PhD students working late in the lab, stressed about their research, questioning their life choices and academia.\n\n<|im_start|>student1\n[Brutally honest observation about academia, research, or life]\n<|im_end|>"
    },
    {
        "name": "Retail Workers During Break",
        "context": "Retail employees in the break room, dealing with difficult customers and corporate policies they can't control.",
        "character_dynamic": "Front-line workers who absorb society's frustrations, finally able to vent without professional smile.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Customer entitlement and social class dynamics",
            "Corporate policies that make no sense on the ground",
            "Being treated as less-than-human by customers",
            "Economic necessity vs personal dignity",
            "Dreams deferred while paying bills",
            "The psychological toll of service work"
        ],
        "prompt_guidelines": "Brutally honest observation about customers, work, or life",
        "characters": [
            "employee1",
            "employee2"
        ],
        "prompt_template": "Retail employees in the break room, dealing with difficult customers and corporate policies they can't control.\n\n<|im_start|>employee1\n[Brutally honest observation about customers, work, or life]\n<|im_end|>"
    },
    {
        "name": "Construction Workers at Lunch",
        "context": "Construction crew eating lunch on a job site, blue-collar perspective and no-nonsense communication style.",
        "character_dynamic": "Physical laborers with practical worldview, direct communication without corporate polish.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Physical toll of manual labor on body and family",
            "Class resentment toward white-collar workers",
            "Pride in building things that last vs desk jobs",
            "Economic insecurity despite essential work",
            "Changing industry and new technologies",
            "Respect earned through competence, not credentials"
        ],
        "prompt_guidelines": "Direct observation about work, class, or life struggles",
        "characters": [
            "worker1",
            "worker2"
        ],
        "prompt_template": "Construction crew eating lunch on a job site, speaking with blue-collar honesty:\n\n<|im_start|>worker1\n[Direct observation about work, class, or life struggles]\n<|im_end|>\n\n<|im_start|>worker2"
    },
    {
        "name": "Dog Park Regulars",
        "context": "Dog owners who see each other daily at the park, bonded by pet ownership and morning routine honesty.",
        "character_dynamic": "People whose dogs have forced them into a social routine, casual familiarity without social pretense.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Using pets to avoid human relationships",
            "Suburban isolation and manufactured community",
            "Pet ownership as substitute for other life goals",
            "Social hierarchies even in dog park dynamics",
            "Spending more on pets than personal needs",
            "What pet behavior reveals about their owners"
        ],
        "prompt_guidelines": "Blunt observation about pets, life, or social dynamics",
        "characters": [
            "owner1",
            "owner2"
        ],
        "prompt_template": "Dog owners at the park, speaking honestly during their daily routine:\n\n<|im_start|>owner1\n[Blunt observation about pets, life, or social dynamics]\n<|im_end|>\n\n<|im_start|>owner2"
    },
    {
        "name": "Night Security Guards",
        "context": "Security guards on overnight shift at an office building, long hours of boredom leading to deep conversations.",
        "character_dynamic": "Workers in an isolating job, darkness and emptiness creating space for unusual honesty.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Existential loneliness of overnight work",
            "Observing empty corporate spaces and questioning purpose",
            "Social isolation and its psychological effects",
            "Economic necessity of taking undesirable jobs",
            "What they think about during long, quiet hours",
            "Being invisible to the daytime world"
        ],
        "prompt_guidelines": "Raw reflection on isolation, work, or existential thoughts",
        "characters": [
            "guard1",
            "guard2"
        ],
        "prompt_template": "Security guards on overnight shift, sharing honest thoughts during the quiet hours:\n\n<|im_start|>guard1\n[Raw reflection on isolation, work, or existential thoughts]\n<|im_end|>\n\n<|im_start|>guard2"
    },
    {
        "name": "Freelancers at Coffee Shop",
        "context": "Freelance workers who've been at the same coffee shop for hours, gig economy struggles and isolation.",
        "character_dynamic": "Independent workers dealing with uncertainty and lack of traditional workplace community.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Financial instability and feast-or-famine cycles",
            "Lack of benefits and social safety net",
            "Isolation from working alone constantly",
            "Imposter syndrome without traditional credentials",
            "Client relationships and power dynamics",
            "The myth of \"being your own boss\""
        ],
        "prompt_guidelines": "Honest struggle about work instability, isolation, or finances",
        "characters": [
            "freelancer1",
            "freelancer2"
        ],
        "prompt_template": "Freelancers who've been working at the same coffee shop for hours, sharing gig economy realities:\n\n<|im_start|>freelancer1\n[Honest struggle about work instability, isolation, or finances]\n<|im_end|>\n\n<|im_start|>freelancer2"
    },
    {
        "name": "Commuters on Delayed Train",
        "context": "Strangers on a delayed train, shared frustration breaking down social barriers, temporary intimacy of shared inconvenience.",
        "character_dynamic": "People whose normal routines are disrupted, creating space for unusual openness with strangers.",
        "generate_prompts": "Generate 20 prompts like:",
        "conversation_topics": [
            "Modern life's dependence on failing infrastructure",
            "Work-life balance and commuting's psychological toll",
            "Social isolation despite physical proximity to others",
            "Economic inequality visible in transportation systems",
            "Environmental guilt about daily choices",
            "The absurdity of modern urban life"
        ],
        "prompt_guidelines": "Frustrated observation about modern life, work, or transportation",
        "characters": [
            "commuter1",
            "commuter2"
        ],
        "prompt_template": "Strangers on a delayed train, shared frustration leading to unexpected honesty:\n\n<|im_start|>commuter1\n[Frustrated observation about modern life, work, or transportation]\n<|im_end|>\n\n<|im_start|>commuter2"
    }
]


def json2prompt(scenario_json):
    """
    Convert a scenario JSON object to the formatted scenario string.
    
    Args:
        scenario_json (dict): Dictionary containing scenario information
        
    Returns:
        str: Formatted scenario string
    """
    name = scenario_json["name"]
    context = scenario_json["context"]
    character_dynamic = scenario_json["character_dynamic"]
    generate_prompts = scenario_json.get("generate_prompts", "Generate 20 prompts like:")
    prompt_template = scenario_json.get("prompt_template", "")
    conversation_topics = scenario_json["conversation_topics"]
    
    # Build the scenario string
    scenario_str = f"""
## Scenario: {name}

**Context:** {context}

**Character Dynamic:** {character_dynamic}

**{generate_prompts}**
```
{prompt_template}
```

**Topics should cover:**
"""
    x3
    # Add conversation topics as bullet points
    for topic in conversation_topics:
        scenario_str += f"- {topic}\n"
    
    return scenario_str


# Test the function with the first scenario
if __name__ == "__main__":
    test_scenario = scenarios_json[0]
    print("Testing json2prompt function:")
    print("="*50)
    print(json2prompt(test_scenario))

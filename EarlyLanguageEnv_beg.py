import gymnasium
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
import pygame
import random

class EarlyLanguageEnvBeg(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
       
        self.child_dictionary = {
                                'year 1' : 
                                    {
                                    'consonants' : ['b', 'd', 'm', 'w', 'g'], 
                                    'vowels' : ['oo', 'uh', 'ah'], 
                                    'target' : ['g', 'oo'], 
                                    'allowed' : 'consonant + vowel'
                                    },                                 
                                'year 2' : 
                                    {
                                    'consonants' : ['b', 'd', 'm', 'w', 'g', 'p', 't', 'n', 'y', 'k'], 
                                    'vowels' : ['oo', 'uh', 'ah', 'ee', 'eh'], 
                                    'target' : ['k', 'oo', 'k', 'oo'], 
                                    'allowed' : 'consonant + vowel + sameConsonant + sameVowel'
                                    },
                                'year 3' : 
                                    {
                                    'consonants' : ['b', 'd', 'm', 'w', 'g', 'p', 't', 'n', 'y', 'k', 'ng', 's', 'z', 'l', 'j', 'r', 'ch', 'zh', 'sh', 'th', 'h', 'f'], 
                                    'vowels' : ['oo', 'uh', 'ah', 'ee', 'eh', 'ih'],
                                    'target' : ['k', 'oo', 'k', 'ee'], 
                                    'allowed' : 'consonant + vowel + consonant + vowel'
                                    }
                                }
        
        self.adult_dictionary = {
                                    'consonants' : ['b', 'd', 'm', 'w', 'g', 'p', 't', 'n', 'y', 'k', 'ng', 's', 'z', 'l', 'j', 'r', 'ch', 'zh', 'sh', 'th', 'h', 'f'], 
                                    'vowels' : ['oo', 'uh', 'ah', 'ee', 'eh', 'ih']
                                }
        max_consonants = len(self.adult_dictionary['consonants']) + 1  # +1 for 'no selection'
        max_vowels = len(self.adult_dictionary['vowels']) + 1  # +1 for 'no selection'
        self.action_space = spaces.MultiDiscrete([max_consonants, max_vowels, max_consonants, max_vowels])

        self.years = ['year 1', 'year 2', 'year 3']
        self.adult_target = ['k', 'oo', 'k', 'ee']
        self.total_dictionary = self.adult_dictionary['consonants'] + self.adult_dictionary['vowels']
        #max_total_indices = len(self.total_dictionary)
        #self.action_space = spaces.MultiDiscrete([max_total_indices, max_total_indices, max_total_indices, max_total_indices])

        self.hour_of_day = 1
        self.day_of_year = 1
        self.year_index = 0
        self.current_year = self.years[self.year_index]
        self.max_response_length = 50

        self.observation_space = spaces.Dict({
            'hour_of_day': spaces.Discrete(24),
            'day_of_year': spaces.Discrete(365),
            'year_index': spaces.Discrete(len(self.years)), 
            'parent_response': spaces.MultiDiscrete([len(self.total_dictionary)] * self.max_response_length)
        })

        # Update current year and its valid phonemes
        self.update_valid_actions()

        self.time_probability = {
             1 : 0.01, 
             2 : 0.01,
             3 : 0.01,
             4 : 0.01,
             5 : 0.01,
             6 : 0.1, # breakfast
             7 : 0.5,
             8 : 1.0,
             9 : 1.0,
            10 : 1.0,
            11 : 0.5,
            12 : 0.1, # lunch
            13 : 0.5, 
            14 : 1.0,
            15 : 1.0,
            16 : 1.0,
            17 : 0.5,
            18 : 0.1, # dinner
            19 : 0.5,
            20 : 1.0,
            21 : 1.0,
            22 : 0.01, # bed time
            23 : 0.01,
            24 : 0.01,
        }

        self.total_steps = 24 * 365 * 3 # total hours in 3 years
        self.step_count = 0
        self.cookie_count = 0
        self.parent_response = 0
        self.parent_sentence = ''
        self.no_action = 0
        self.wrong_guess = 0
        self.total_reward = 0
        self.done = False

        self.child_state = 'rest' # ask
        self.parent_state = 'rest' # respond, sleep, eat, cookie

    def step(self, action):

        phonemes = self.interpret_phonemes(action)
        self.step_count += 1

        self.hour_of_day += 1
        if self.hour_of_day > 24:
            self.hour_of_day = 1
            self.day_of_year += 1

        if self.day_of_year > 365:
            self.day_of_year = 1
            self.year_index = (self.year_index + 1) % len(self.years)

        if self.step_count >= self.total_steps:
            self.done = True
            observation = {
                'hour_of_day': self.hour_of_day,
                'day_of_year': self.day_of_year,
                'year_index': self.year_index, 
                'parent_response' : []
            }
            return observation, self.total_reward, self.done, self.done, {'Total Calculated Reward': self.total_reward,
                                                        'Cookie Count': self.cookie_count,
                                                        'Parent Responses': self.parent_response,
                                                        'Parent Sentence': '',
                                                        'Child Sentence': '', 
                                                        'No Action Steps': self.no_action,
                                                        'Wrong Guesses': self.wrong_guess, 
                                                        'Year': self.current_year}
        
        self.current_year = self.years[self.year_index]
        self.update_valid_actions()

        if self.time_probability[self.hour_of_day] == 0.01:
            self.parent_state = 'sleep'
        elif self.time_probability[self.hour_of_day] == 0.1:
            self.parent_state = 'eat'
        else:
            self.parent_state = 'rest'
        
        sentence = ''

        if not phonemes:  # Checking if the list is empty (no action taken)
            self.child_state = 'rest'
            self.no_action += 1
        else:
            self.child_state = 'ask'
            target = self.child_dictionary[self.current_year]['target']
            response_probability = self.time_probability[self.hour_of_day]
            parent_will_respond = random.random() < response_probability
            correct_guess = all(elem in target for elem in phonemes)
            partial_guess = any(phoneme == target[i] for i, phoneme in enumerate(phonemes) if i < len(target))

            if parent_will_respond and correct_guess: 
                self.parent_state = 'cookie'
                self.child_state = 'cookie'
                sentence = self.generate_sentence_based_on_year()
                self.cookie_count += 1
            elif parent_will_respond and partial_guess:
                self.parent_state = 'respond'
                sentence = self.generate_sentence_based_on_year()
                self.parent_response += 1
            elif not correct_guess and not partial_guess:
                self.wrong_guess += 1
            elif not parent_will_respond and correct_guess:
                self.parent_response += 0.5            
            elif not parent_will_respond and partial_guess:
                self.parent_response += 0.3
            

        self.total_reward = 1*self.cookie_count + 0.15*self.parent_response - 0.05* self.wrong_guess - 0.075* self.no_action

        observation = {
            'hour_of_day': self.hour_of_day,
            'day_of_year': self.day_of_year,
            'year_index': self.year_index, 
            'parent_response': sentence
        }
        # print('Observation: ', observation, '; Phonemes: ', phonemes)
        return observation, self.total_reward, self.done, self.done, {'Total Calculated Reward': self.total_reward,
                                                        'Cookie Count': self.cookie_count,
                                                        'Parent Responses': self.parent_response,
                                                        'Parent Sentence': sentence,
                                                        'Child Sentence' : phonemes, 
                                                        'No Action Steps': self.no_action,
                                                        'Wrong Guesses': self.wrong_guess, 
                                                        'Year': self.current_year}

    def interpret_phonemes(self, action):
        # Initialize an empty list for phonemes
        phonemes = []

        # Determine the action length based on the current year
        action_length = len(action)

        # Iterate over the action array
        for i in range(action_length):
            if 0 < action[i] <= len(self.total_dictionary):
                # Valid index, subtract 1 since action is 1-indexed
                index = action[i] - 1
                if i % 2 == 0:  # If the index is even
                    phoneme = self.adult_dictionary['consonants'][index]
                else:  # If the index is odd
                    phoneme = self.adult_dictionary['vowels'][index]
                phonemes.append(phoneme)

        # Ensure the correct format for each year
        if self.current_year == 'year 1':
            return phonemes[:2]  # Only consider the first consonant and vowel
        elif self.current_year == 'year 2':
            return phonemes[:2] * 2 if all(phonemes[:2]) else []  # Duplicate if valid
        elif self.current_year == 'year 3':
            return phonemes[:4]  # Consider all four phonemes

        return phonemes

    
    def insert_target_randomly(self, sentence, target):
        target_str = ''.join(target)
        sentence_str = ''.join([''.join(pair) for pair in sentence])
        insert_position = random.randint(0, len(sentence_str) - len(target_str))
        new_sentence_str = sentence_str[:insert_position] + target_str + sentence_str[insert_position:]
        new_sentence = [new_sentence_str[i:i+2] for i in range(0, len(new_sentence_str), 2)]
        return new_sentence

    def generate_sentence_based_on_year(self):
        year = self.current_year
        consonants = self.child_dictionary[year]['consonants']
        vowels = self.child_dictionary[year]['vowels']
        target = self.child_dictionary[year]['target']
        sentence = []

        if year == 'year 1':
            for _ in range(15):  # 30 entries, each entry is 2 characters long
                sentence.append([random.choice(consonants), random.choice(vowels)])

        elif year == 'year 2':
            for _ in range(20):  # 40 entries, each entry is 2 characters long
                cv_pair = [random.choice(consonants), random.choice(vowels)]
                sentence.append(cv_pair if random.choice([True, False]) else cv_pair + cv_pair)

        elif year == 'year 3':
            for _ in range(12):  # 48 entries, each entry is 4 characters long
                sentence.append([random.choice(consonants), random.choice(vowels)])
                sentence.append([random.choice(consonants), random.choice(vowels)])

        return self.insert_target_randomly(sentence, target)

    def update_valid_actions(self):
        # Update the list of valid phonemes based on the current year
        self.valid_phonemes = self.child_dictionary[self.current_year]['consonants'] + self.child_dictionary[self.current_year]['vowels']
        
        # Update action space size (optional, mainly for internal logic)
        self.action_space.n = len(self.valid_phonemes)

    def reset(self, seed=None):
        # Reset the time variables
        self.hour_of_day = 1
        self.day_of_year = 1
        self.year_index = 0
        self.cookie_count = 0
        self.done = False
        self.current_year = self.years[self.year_index]
        self.parent_response = 0
        self.no_action = 0
        self.wrong_guess = 0
        self.total_reward = 0
        self.step_count = 0
        self.update_valid_actions()
        seed = self.seed(seed)
        observation = {
            'hour_of_day': self.hour_of_day,
            'day_of_year': self.day_of_year,
            'year_index': self.year_index, 
            'parent_response' : []
        }
        return observation, {}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        # Initialize Pygame if it hasn't been initialized yet
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
        
        # Clear the screen
        self.screen.fill((255, 255, 255))  # White background
        
        # Draw state-specific symbols with appropriate padding
        parent_image_pos = (550, 270)  # Right side of the parent
        child_image_pos = (200, 270)    # Left side of the child

        if self.parent_state == 'sleep':
            self.draw_sheep(self.screen, parent_image_pos)
        elif self.parent_state == 'eat':
            self.draw_apple(self.screen, parent_image_pos)
        elif self.parent_state == 'respond':
            self.draw_chat_bubble(self.screen, parent_image_pos)
        elif self.parent_state == 'cookie':
            self.draw_cookie(self.screen, parent_image_pos)

        if self.child_state == 'ask':
            self.draw_chat_bubble(self.screen, child_image_pos)
        elif self.child_state == 'cookie':
            self.draw_cookie(self.screen, child_image_pos)

        # Draw the parent and child
        self.draw_parent(self.screen, (0, 0, 0))
        self.draw_child(self.screen, (0, 0, 0))

        # Fonts
        main_font = pygame.font.Font(None, 36)
        sub_font = pygame.font.Font(None, 24)  # Smaller font for sub-scores

        # Render main labels and total score
        parent_label = main_font.render('Parent', True, (0, 0, 0))
        child_label = main_font.render('Child', True, (0, 0, 0))
        reward_text = main_font.render(f'Total Score: {self.total_reward:.2f}', True, (0, 0, 0))

        # Centering the labels over the heads
        parent_head_center = (700, 270)
        child_head_center = (100, 270)
        parent_label_pos = (parent_head_center[0] - parent_label.get_width() // 2, parent_head_center[1] - 80)
        child_label_pos = (child_head_center[0] - child_label.get_width() // 2, child_head_center[1] - 60)

        # Positioning main labels and total score
        self.screen.blit(parent_label, parent_label_pos)
        self.screen.blit(child_label, child_label_pos)
        self.screen.blit(reward_text, (350, 550))  # Bottom center

        # Display subreward/punishment scores
        score_text = sub_font.render(f'Parent Response: {self.parent_response:.2f}', True, (0, 0, 0))
        no_action_text = sub_font.render(f'No Action: {self.no_action:.2f}', True, (0, 0, 0))
        wrong_guess_text = sub_font.render(f'Wrong Guesses: {self.wrong_guess:.2f}', True, (0, 0, 0))
        cookie_count_text = sub_font.render(f'Cookie Count: {self.cookie_count:.2f}', True, (0, 0, 0))

        # Positioning subreward/punishment scores
        self.screen.blit(score_text, (550, 10))     # Top right corner
        self.screen.blit(no_action_text, (550, 40))
        self.screen.blit(wrong_guess_text, (550, 70))
        self.screen.blit(cookie_count_text, (550, 100))

        # Update the display
        pygame.display.flip()

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
    
    def draw_parent(self, screen, parent_color):
        # Drawing a simple humanoid figure for the parent
        pygame.draw.circle(screen, parent_color, (700, 270), 30)  # Head
        pygame.draw.line(screen, parent_color, (700, 300), (700, 400), 5)  # Body
        pygame.draw.line(screen, parent_color, (700, 320), (660, 360), 5)  # Left arm
        pygame.draw.line(screen, parent_color, (700, 320), (740, 360), 5)  # Right arm
        pygame.draw.line(screen, parent_color, (700, 400), (680, 450), 5)  # Left leg
        pygame.draw.line(screen, parent_color, (700, 400), (720, 450), 5)  # Right leg

    def draw_child(self, screen, child_color):
        # Drawing a simple humanoid figure for the child
        pygame.draw.circle(screen, child_color, (100, 270), 20)  # Head
        pygame.draw.line(screen, child_color, (100, 290), (100, 350), 3)  # Body
        pygame.draw.line(screen, child_color, (100, 300), (80, 320), 3)  # Left arm
        pygame.draw.line(screen, child_color, (100, 300), (120, 320), 3)  # Right arm
        pygame.draw.line(screen, child_color, (100, 350), (90, 380), 3)  # Left leg
        pygame.draw.line(screen, child_color, (100, 350), (110, 380), 3)  # Right leg

    def draw_sheep(self, screen, position):
        # Drawing a sheep with a slightly gray coat and one eye
        pygame.draw.circle(screen, (220, 220, 220), position, 20)  # Sheep body (slightly gray)
        pygame.draw.circle(screen, (220, 220, 220), (position[0] + 20, position[1]), 10)  # Sheep head
        # Eye
        pygame.draw.circle(screen, (0, 0, 0), (position[0] + 22, position[1] - 3), 2)
        # Legs
        pygame.draw.line(screen, (0, 0, 0), (position[0] - 10, position[1] + 15), (position[0] - 10, position[1] + 25), 2)
        pygame.draw.line(screen, (0, 0, 0), (position[0] + 10, position[1] + 15), (position[0] + 10, position[1] + 25), 2)
        # Zzz's next to the sheep
        small_font = pygame.font.Font(None, 24)
        zzz_text = small_font.render('zzz', True, (0, 0, 255))
        screen.blit(zzz_text, (position[0] + 30, position[1] - 20))

    def draw_cookie(self, screen, position):
        # Drawing a cookie with chocolate chips
        pygame.draw.circle(screen, (210, 105, 30), position, 15)  # Cookie
        # Chocolate chips
        pygame.draw.circle(screen, (139, 69, 19), (position[0] + 5, position[1] + 5), 3)
        pygame.draw.circle(screen, (139, 69, 19), (position[0] - 5, position[1] - 5), 3)
        pygame.draw.circle(screen, (139, 69, 19), (position[0] - 5, position[1] + 5), 2)
        pygame.draw.circle(screen, (139, 69, 19), (position[0] + 5, position[1] - 5), 2)

    def draw_apple(self, screen, position):
        # Drawing an apple with a stem and leaf
        pygame.draw.circle(screen, (255, 0, 0), position, 15)  # Apple
        # Stem
        pygame.draw.line(screen, (139, 69, 19), (position[0], position[1] - 15), (position[0], position[1] - 20), 2)
        # Leaf
        leaf = [(position[0] + 5, position[1] - 20), (position[0] + 10, position[1] - 18), (position[0] + 5, position[1] - 15)]
        pygame.draw.polygon(screen, (34, 139, 34), leaf)

    def draw_chat_bubble(self, screen, position):
        # Drawing a chat bubble with a black outline
        # Black outline
        pygame.draw.circle(screen, (0, 0, 0), position, 22)  # Slightly larger black circle
        # White bubble
        pygame.draw.circle(screen, (255, 255, 255), position, 20)  # White inner circle
        
        small_font = pygame.font.Font(None, 20)
        text = small_font.render('?!', True, (0, 0, 0))
        screen.blit(text, (position[0] - text.get_width() // 2, position[1] - text.get_height() // 2))
    

    def close(self):
        # Check if Pygame was initialized and if the screen exists
        if hasattr(self, 'screen'):
            # Quit Pygame
            pygame.quit()

            # Delete the screen attribute to clean up
            del self.screen

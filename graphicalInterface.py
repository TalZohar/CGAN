import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pygame
import glob
import numpy as np
from tensorflow.keras.models import load_model


def greyscale(surface: pygame.Surface):
    arr = pygame.surfarray.array3d(surface)
    # calulates the avg of the "rgb" values, this reduces the dim by 1
    mean_arr = np.mean(arr, axis=2)
    # restores the dimension from 2 to 3
    mean_arr3d = mean_arr[..., np.newaxis]
    # repeat the avg value obtained before over the axis 2
    new_arr = np.repeat(mean_arr3d[:, :, :], 3, axis=2)
    # return the new surface
    return pygame.surfarray.make_surface(new_arr)


def getGeneratedImage(latent_points, categories, size):
    latent_points = latent_points.reshape((1,len(latent_points)))
    categories = categories.reshape((1,len(categories)))

    image = model.predict([(latent_points), (categories)])
    image = (image[0] + 1)*125.5
    image = pygame.surfarray.make_surface(image)
    image = pygame.transform.rotate(image, -90)
    return pygame.transform.scale(image, (size, size))


class Button(object):

    def __init__(self, position, size, image, type = True):

        # create 2 images
        self.images = [
            pygame.Surface(size),
            pygame.Surface(size),
        ]
        self.type = type

        # fill images with color - red, gree, blue
        self.images[0] = pygame.transform.scale(greyscale(image), size)
        self.images[1] = pygame.transform.scale(image, size)

        # get image size and position
        self.rect = pygame.Rect(position, size)
        self.index = 0

    def draw(self, screen):
        # draw selected image
        screen.blit(self.images[self.index], self.rect)

    def event_handler(self, event):
        # change selected color if rectange clicked
        if event.type == pygame.MOUSEBUTTONDOWN:  # is some button clicked
            if event.button == 1:  # is left button clicked
                if self.rect.collidepoint(event.pos):  # is mouse over button
                    save = self.index
                    if not pygame.key.get_mods() & pygame.KMOD_CTRL and self.type:
                        for button in buttons:
                            button.index = 0
                    self.index = (save + 1) % 2  # change image
                    return True
        return False


# --- main ---

WIDTH = 1000
HEIGHT = 600
image_size = 128

# init

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# create buttons

paths = glob.glob(r'Dataset\thumbnails\*.png')
button_size = WIDTH // len(paths)
buttons = [Button((n * button_size, 5), (button_size, button_size), pygame.image.load(path)) for n, path in
           enumerate(paths)]
stop_button = Button((0, HEIGHT-button_size), (button_size, button_size), pygame.image.load(r'Dataset\thumbnails\play.jpg'), False)

# model
model = load_model("saved_model/g_model30.h5")
latent_dim = 128
latent_points = np.random.normal(size = latent_dim)

# mainloop
running = True
active_categories = np.array([button.index for button in buttons])

while running:
    screen.fill((0, 0, 0))
    # --- events ---
    num_categories = np.sum(np.array([button.index for button in buttons]))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # --- buttons events ---
        for button in buttons:
            if button.event_handler(event):
                if num_categories:
                    active_categories = np.array([button.index for button in buttons])
        stop_button.event_handler(event)

        #--reroll
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                latent_points = np.random.normal(size=latent_dim)
        #--save
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                print(latent_points)

    #update latent points
    if stop_button.index == 1:
        latent_points[np.random.randint(latent_dim)] = np.random.normal()
    # --- draws ---
    for button in buttons:
        button.draw(screen)
    stop_button.draw(screen)
    if num_categories:
        screen.blit(getGeneratedImage(latent_points, active_categories, image_size).convert_alpha(), ((WIDTH-image_size)/2,HEIGHT/2))

    pygame.display.update()

# --- the end ---

pygame.quit()

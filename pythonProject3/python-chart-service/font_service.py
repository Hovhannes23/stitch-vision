from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont


def get_fonts_from_file(path):

    # sample text and font
    unicode_text = u"Hello World!"
    path = "/home/user/PycharmProjects/pythonProject3/python-chart-service/fonts/CSTITCHD.ttf"

    font_TTFont = TTFont(path)
    glyph_set = TTFont.getGlyphSet(font_TTFont)
    glyphs = glyph_set.glyfTable.glyphs



    font = ImageFont.truetype(path, 28, encoding="unic")

    # get the line size
    text_width, text_height = font_TTFont.getsize(unicode_text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', (text_width + 10, text_height + 10), "orange")

    # draw the text onto the text canvas, and use blue as the text color
    draw = ImageDraw.Draw(canvas)
    draw.text((5,5), u'Hello World!', 'blue', font_TTFont)

    # save the blank canvas to a file
    canvas.save("unicode-text.png", "PNG")
    canvas.show()
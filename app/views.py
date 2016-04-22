from flask import render_template, flash, redirect, request
from .forms import AdditionForm, TextGenForm
from app import app
from app.model.textgen import LiteTextGen
import sys

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	form = TextGenForm()
	ltg = LiteTextGen()
	primer_txt = ''

	if form.validate_on_submit():

		primer_txt = form.primer.data
		h = ltg.predict(primer = primer_txt, length = 50, stream=False, diversity = .2)
		return render_template('index.html', title='Text Generation', form=form, prediction=h, primer=primer_txt)
	return render_template('index.html', title='Text Generation', form=form, prediction=None, primer=primer_txt)
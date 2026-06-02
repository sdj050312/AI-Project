from flask import Blueprint, render_template

japan_bp = Blueprint('japan', __name__)

@japan_bp.route('/japan')
def japan():

    return render_template('japan.html')
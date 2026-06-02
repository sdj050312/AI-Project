from flask import Blueprint, render_template

korea_bp = Blueprint('korea', __name__)

@korea_bp.route('/korea')
def korea():

    return render_template('korea.html')
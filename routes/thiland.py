from flask import Blueprint, render_template

th_bp = Blueprint('thiland', __name__)

@th_bp.route('/thiland')
def thiland():

    return render_template('thiland.html')  
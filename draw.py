def L_hinge(t):
    result = []
    for item in t :
        result.append(max(0 , 1-item))
    return result
def L_huber_hinge(items , h):
    result = []
    for t in items :
        if t < 1-h:
            result.append( 1-t)
        elif t> 1+h:
            result.append( 0)
        else :
            result.append (((1+h-t)**2)/(4*h))
    return result
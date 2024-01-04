def delete_annotation(annotation,description_or_list_of_desc):
    del_anno = []
    for i in range(len(annotation)):
        if isinstance(description_or_list_of_desc, list):
            if annotation[i]['description'] in description_or_list_of_desc:  # !=event_descripion:
                del_anno.append(i)
        else:
            if annotation[i]['description'] == description_or_list_of_desc:  # !=event_descripion:
                del_anno.append(i)

    annotation.delete(del_anno)  # 124
    return annotation

def keep_annotation(annotation, description_or_list_of_desc):
    del_anno = []
    for i in range(len(annotation)):
        if isinstance(description_or_list_of_desc, list):
            if annotation[i]['description'] not in description_or_list_of_desc:  # !=event_descripion:
                del_anno.append(i)
        else:
            if annotation[i]['description'] != description_or_list_of_desc:  # !=event_descripion:
                del_anno.append(i)

    annotation.delete(del_anno)  # 124
    return annotation

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

# delete annotation from annotations which is not or not contained in the description_or_list_of_desc.
def keep_annotation(annotations, description_or_list_of_desc):
    del_anno = []
    for i in range(len(annotations)):
        if isinstance(description_or_list_of_desc, list):
            if annotations[i]['description'] not in description_or_list_of_desc:  # !=event_descripion:
                del_anno.append(i)
        else:
            if annotations[i]['description'] != description_or_list_of_desc:  # !=event_descripion:
                del_anno.append(i)

    annotations.delete(del_anno)  # 124
    return annotations

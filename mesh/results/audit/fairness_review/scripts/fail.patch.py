def patch(code):
    old = 'record["qaoa"] = run_qaoa_groups(inst, key)'
    assert code.count(old) == 1
    new = ('import os as _os\n                if _os.environ.get("INJECT_FAIL") and key[0] == "9 priority" and key[3] == "2 equal devices":\n'
           '                    raise RuntimeError("injected reviewer failure")\n                record["qaoa"] = run_qaoa_groups(inst, key)')
    return code.replace(old, new)

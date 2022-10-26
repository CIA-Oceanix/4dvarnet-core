import utils
import traceback


base_cfg = 'baseline/full_core'
fp = 'dgx_ifremer'
overrides = [
    'file_paths={fp}'
]



def run1():
    try:

        cfg = utils.get_cfg(base_cfg)
        dm = utils.get_dm(base_cfg, add_overrides=overrides)
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


def main():
    try:
        fn = run1

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    main()
